import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

# Marcel Haas, Nov 2022 (datascience@marcelhaas.com)

   
def plot_example_grades():
    plt.figure(figsize=(12,5))
    plt.subplot(121)

    ability = 0
    difficulty = np.linspace(-5,5, num=100)

    min_grade = 0
    max_grade = 100
    
    c = grade(ability, difficulty, min_grade=min_grade, max_grade=max_grade)
    plt.plot(difficulty, c, label="Ability = 0", linewidth=8)

    ability = 3
    c = grade(ability, difficulty, min_grade=min_grade, max_grade=max_grade)
    plt.plot(difficulty, c, label="Ability = 3", linewidth=8)
    plt.xlabel("Difficulty", fontsize=18)
    plt.ylabel("Estimated grade", fontsize=18)
    plt.legend(fontsize=14)
    ########################################
    plt.subplot(122)
    difficulty = 0
    ability = np.linspace(-5,5, num=100)

    c = grade(ability, difficulty, min_grade=min_grade, max_grade=max_grade)
    plt.plot(ability, c, label="Difficulty = 0", linewidth=8)

    difficulty = 3
    c = grade(ability, difficulty, min_grade=min_grade, max_grade=max_grade)
    plt.plot(ability, c, label="Difficulty = 3", linewidth=8)
    plt.xlabel("Ability", fontsize=18)
    plt.ylabel("Estimated grade", fontsize=18)
    plt.legend(fontsize=14);
    return

def grade(ability, difficulty, disc=1, noise=None, min_grade=6, max_grade=10, rounding=0):
    if isinstance(ability, np.ndarray):
        mean_ab = ability.mean()
    else: mean_ab = 0
    cijfer = (max_grade - min_grade) / (1+ np.exp(disc*(difficulty - (ability-mean_ab)))) + min_grade
    if noise: cijfer += np.random.normal(0, noise, size=cijfer.shape)
    return np.round(cijfer, rounding)


def create_data(difficulties=None, ability_diff=3., ability_std=0.1, n_st=500,
               noise_cijfers=0.01):
    """ Simple function to set difficulties and abilities for experiment.
    If none are given, the setup of the first simple experiment is used. 
    Other possibilities:
    difficulties: list/array of difficulties
    ability_diff: int/float that indicates the difference in mean ability between the groups
    ability_std: float: standard deviation of normally distributed ability with the groups 
                        (same for both groups)
    n_st: int: number of students in each group
    noise_cijfers: float: (gaussian) noise on grades
 
    returns:
    grades: array of grades, in which everybody did every course
    """
    
    if not difficulties:
        diff_dom = np.array([-1.5, -1., -0.5])
        diff_slim = diff_dom + 3
        diff_link = np.array([0.5])

        difficulties = np.concatenate((diff_dom, diff_link, diff_slim))
    difficulties = np.array(difficulties).reshape(1,-1)

    np.random.seed(0)

    n_st = 500

    ability_dom = 0.
    
    abilities_dom = np.random.normal(loc=ability_dom, scale=ability_std, size=n_st)
    abilities_slim = np.random.normal(loc=ability_dom+ability_diff, scale=ability_std, size=n_st)
    abilities = np.concatenate((abilities_dom, abilities_slim)).reshape(-1, 1)

    grades = grade(abilities, difficulties, noise=noise_cijfers)
    
    # Set to NaN for those who didn't do some courses:
    grades[n_st:,:3] = np.nan
    grades[:n_st,4:] = np.nan
    
    return grades

 
def grade_df(grades, 
              course_names=['Course A', 'Course B', 'Course C', 'Course X', 
                            'Course D', 'Course E', 'Course F']):
    
    """ Utility function to morph np.array into dataframe and melt into long form.
    
    Apologies for the Dutch variable names :)
    """
    # To df
    alle_cijfers = pd.DataFrame(data=grades, columns=course_names).reset_index()
    # To long form
    cijfertjes = pd.melt(alle_cijfers, id_vars=['index']).dropna()
    # Names that mean something
    cijfertjes.rename(
        columns={'index':'StudentNumber', 'variable':'Course', 'value':'Grade'}, 
        inplace=True)
    
    return cijfertjes

def model_and_visualize(grades_df):
    
    # Some data structures we need
    courses = grades_df.Course.unique(); n_courses = len(courses)
    course_enum = {v:i for i, v in enumerate(courses)}
    course_idx = np.array([course_enum[v] for v in grades_df.Course])

    students = grades_df.StudentNumber.unique(); n_students = len(students)
    student_enum = {s:i for i, s in enumerate(students)}
    student_idx = np.array([student_enum[s] for s in grades_df.StudentNumber])

    # A pooled, hierarchical model for courses:
    with pm.Model() as simulatie:
        # Property of the courses
        δ = pm.Normal('Course difficulty', 0, 3, shape=n_courses)
        
        # Properties of the students: ability
        α = pm.Normal('Student ability', 0, 3, shape=n_students)
        
        # Uncertainty in grade follows a Gaussian with stdev:
        ϵ = pm.HalfNormal('eps', 10)
        
        # Estimated grade from ability and difficulty
        grade_estimate = 100 / (1+np.exp(δ[course_idx] - (α[student_idx]-α.mean() )))
        # Likelihood is that gaussian
        grades = pm.Normal("Grades", grade_estimate, sigma=ϵ, observed=grades_df.Grade)

        # InferenceButton(TM)
        step = pm.NUTS()
        trace = pm.sample(1000, cores=4, step=step, tune=2000)

    az.plot_trace(trace, var_names=['Course difficulty', 'eps'], figsize=(10,6));
    
    return trace, simulatie
    
def course_network(df, 
                   course_list=None, 
                   min_students_course=100, 
                   min_students_overlap=10,
                  ):
    """Build a network of courses with metadata, based
    on students taking the course.
    
    Inputs:
    -------
    df: 
        DataFrame with studentnummer-vakcode combinations
        to see who took what. Cijfer are used for network 
        metadata, if included.
    
    Optional:
    ---------
    course_list: 
        list of courses to take into account. If not provided,
        all courses in df are taken.
    min_students_course: 
        minimum number of students per course to be 
        included in the network. Default=100
    min_students_overlap:
        minimum number of students that overlap between two courses
        to include the edge in the network. Default=10
                 
    Output: Graph of courses with:
            - Nodes are courses, metadata: 
              {nstudents: how many students took the course,
               av_grade: average of grades for these courses}
            - Edges are groups of students who took both node courses
              Metadata: {nstudents: number of common students}
    """
    
    # Build original network of student-course combis
    G = nx.from_pandas_edgelist(df, 'StudentNumber', 'Course')
    
    # Just for safety:
    Cijfer = 'Grade' in df
    if not Cijfer: 
        print('No Grade in DataFrame, no Grade information in output')
    
    # Compile course metadata
    percourse = df.groupby('Course')
    n_students = percourse.StudentNumber.nunique()
    nst = pd.DataFrame(n_students).rename(
        columns={'StudentNumber':'Number'})
    if Cijfer: av_grades = percourse.Grade.mean()
    df = df.merge(nst, how='inner', left_on="Course", right_index=True)
    df = df[df.Number >= min_students_course]
    
    # Create course list if not provided
    if not course_list: course_list = df.Course.unique()
        
    # Loop over courses and use data from G to build course network
    course_net = nx.Graph()
    # Loop over all courses ...
    for ic, c in enumerate(course_list): 
        ns_course = n_students[c]
        if ns_course < min_students_course: continue
        # And check for all other courses ...
        for c2 in course_list[ic+1:]:
            # How many students there are in common
            common_students = len(list(nx.common_neighbors(G, c, c2)))
            # Add edge (end perhaps nodes) with data on nr of students
            if common_students >= min_students_overlap:
                course_net.add_edge(c, c2, **{'nstudents':common_students})
            else:
                # Want to still add the node, even though no connections
                course_net.add_nodes_from([c, c2])
        
        # add course metadata to network
        course_net.nodes[c]['nstudents'] = ns_course
        if Cijfer: course_net.nodes[c]['avgrade'] = av_grades[c]
        
    # Remove unconnected nodes
    course_net.remove_nodes_from(list(nx.isolates(course_net)))
    
    return course_net


def show_course_network(course_net, use_metadata=True, kind='spring'):
    """Simple graph plotting function, just to save spcae in the notebook.
    
    Takes a graph and uses some pre-defined metadata to plot a network.    
    """
    
    
    # Aux variables for plot normalisation
    nodes_data = np.array([v[1]['nstudents'] for v in course_net.nodes.data()] )
    min_ns, max_ns = nodes_data.min(), nodes_data.max()
    edges_data = np.array([v[2]['nstudents'] for v in course_net.edges.data()] )
    min_no, max_no = edges_data.min(), edges_data.max()
    
    fig, ax = plt.subplots(figsize=(20, 15))
    if kind == 'spring':
        pos = nx.spring_layout(course_net, k=0.7, seed=42) 
    elif kind == 'shell':
        pos = nx.shell_layout(course_net, )
    elif kind == 'spiral':
        pos = nx.spiral_layout(course_net)
    elif kind == 'spectral':
        pos = nx.spectral_layout(course_net)
    elif kind == "kamada_kawai":
        pos = nx.kamada_kawai_layout(course_net)
    elif kind == "planar":
        pos = nx.planar_layout(course_net)
    else:
        print("Kind of graph not recognized; known: (spring, shell," 
              "spiral, spectral, kamada_kawai, planar), exiting...")
        return
    try:
        node_size = [(v[1]['nstudents']-min_ns+1)*5000/(max_ns - min_ns) for v in course_net.nodes.data()]
        node_color = [v[1]['avgrade'] for v in course_net.nodes.data()]
        widths = [3*(v[2]['nstudents']-min_no)/(max_no - min_no)+.1 for v in course_net.edges.data()]
        nx.draw_networkx(
            course_net,
            pos=pos,
            with_labels=True,
            node_color=node_color,
            node_size=node_size,
            edge_color="black",
            width=widths,
            cmap='viridis',
            alpha=0.9
        )
        font = {"color": "r", "fontweight": "bold", "fontsize": 14}
        plt.title(f"Node size correpsonds to number of students, color to average grade, edge thickness to overlap.", fontdict=font)
        plt.xlabel(f"{nx.number_connected_components(course_net)} separate connected subgraphs", fontdict=font)
        
    except Exception as inst:
        print(type(inst)) 
