#!/usr/bin/env python3
import os

import numpy as np

import main_cl
import options
import utils
from data.HITS import HITSDataset
from param_stamp import get_param_stamp_from_args
from visual import plt


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False, 'generative': True, 'compare_code': 'all'}
    # Define input options
    parser = options.define_args(filename="_compare_HITSsrc",
                                 description='Compare performance of various continual learning strategies on different'
                                             ' settings of HITSsrc domain-incremental learning.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_HITS_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_bir_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    # Add an argument to regenerate pdfs
    parser.add_argument('--regenerate', action='store_true', 
                        help='Regenerate individual pdfs even if they already exist.')
    # Parse and process (i.e., set defaults for unselected options) options
    args = parser.parse_args()
    # HITSsrc benchmark is fixed: Domain-IL scenario.
    assert args.base_tasks in HITSDataset.BASE_HITS_TASK_SRC.keys(), f"For HITS comparison, please precise base tasks (one of: {list(HITSDataset.BASE_HITS_TASK_SRC.keys())})"
    args.experiment = args.base_tasks
    args.scenario = "domain"
    args.tasks = len(HITSDataset.BASE_HITS_TASK_SRC[args.base_tasks].get('task_list', []))
    options.set_defaults(args, **kwargs)
    return args



def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)

    # -check whether already run, and if not do so
    if os.path.isfile("{}/dict-{}.pkl".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
        # Regenerate individual pdfs if requested by user
        if args.regenerate:
            args.train=False
            print("{}: ...regenerating pdf...".format(param_stamp))
            main_cl.run(args)
    else:
        print("{}: ...running...".format(param_stamp))
        main_cl.run(args)
    # -get average accuracies
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -results-dict
    dict = utils.load_object("{}/dict-{}".format(args.r_dir,  param_stamp))
    # -return tuple with the results
    return (dict, ave)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        print(f"\nRunning with seed {seed}...")
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict



if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    # If user specified methods to run, filter the list of methods accordingly
    valid_methods = {"none", "joint", "generative", "lwf", "bir", "ewc", "si", "xdg"}
    if args.methods is not None:
        for method in args.methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid method specified: {method}. Valid options are: {valid_methods}")
        print("Running only specified methods: {}".format(args.methods))
    else:
        print("Running all methods.")
        args.methods = list(valid_methods)
    results_dict = {}

    ###----"BASELINES"----###

    ## None
    if "none" in args.methods:
        args.replay = "none"
        args.distill = False
        NONE = {}
        NONE = collect_all(NONE, seed_list, args, name="None")
        results_dict["none"] = NONE
        

    ## Joint
    if "joint" in args.methods:
        args.replay = "offline"
        OFF = {}
        OFF = collect_all(OFF, seed_list, args, name="Joint")
        results_dict["joint"] = OFF


    ###----"REPLAY"----###

    ## Generative replay
    if "generative" in args.methods:
        args.replay = "generative"
        GR = {}
        GR = collect_all(GR, seed_list, args, name="Generative Replay (GR)")
        results_dict["generative"] = GR

    ## LwF
    if "lwf" in args.methods:
        args.replay = "current"
        args.distill = True
        LWF = {}
        LWF = collect_all(LWF, seed_list, args, name="Learning without Forgetting (LwF)")
        args.replay = "none"
        args.distill = False
        results_dict["lwf"] = LWF

    ## Brain-Inspired Replay (BI-R)
    if "bir" in args.methods:
        args.replay = "generative"
        args.brain_inspired = True
        # Mirror what --brain-inspired configures in main_cl/options.
        args.feedback = True
        args.prior = "GMM"
        args.per_class = True
        args.dg_gates = True
        args.hidden = True
        args.pre_convE = True
        args.freeze_convE = True
        args.distill = True
        BIR = {}
        BIR = collect_all(BIR, seed_list, args, name="Brain-Inspired Replay (BI-R)")
        results_dict["bir"] = BIR
    args.brain_inspired = False
    args.feedback = False
    args.prior = "standard"
    args.per_class = False
    args.dg_gates = False
    args.hidden = False
    args.pre_convE = False
    args.freeze_convE = False
    args.replay = "none"
    args.distill = False


    ###----"EWC / SI"----####

    ## EWC
    if "ewc" in args.methods:
        args.ewc = True
        EWC = {}
        EWC = collect_all(EWC, seed_list, args, name="Elastic Weight Consolidation (EWC)")
        args.ewc = False
        results_dict["ewc"] = EWC

    ## SI
    if "si" in args.methods:
        args.si = True
        SI = {}
        SI = collect_all(SI, seed_list, args, name="Synaptic Intelligence (SI)")
        args.si = False
        results_dict["si"] = SI


    ###----"XdG"----####
    if "xdg" in args.methods:
        if args.scenario=="task":
            args.xdg = True
            XDG = {}
            XDG = collect_all(XDG, seed_list, args, name="Context-dependent Gating (XdG)")
            args.xdg = False
            results_dict["xdg"] = XDG


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    acc = {}
    ave_acc = {}

    ## Create lists for all extracted <dicts> and <lists> with fixed order
    for seed in seed_list:

        i = 0
        acc[seed] = {
            method_name: results_dict[method_name][seed][i]["average"] for method_name in args.methods
        }
        i = 1
        ave_acc[seed] = {
            method_name: results_dict[method_name][seed][i] for method_name in args.methods
        }
        if args.scenario=="task":
            acc[seed]["xdg"] = [results_dict["xdg"][seed][0]["average"]]
            ave_acc[seed]["xdg"] = [results_dict["xdg"][seed][1]]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}{}-{}-{}".format(args.experiment, args.tasks, args.scenario, '-'.join(map(str, args.task_order)))
    scheme = "{}-incremental learning".format(args.scenario)
    title = "{}   -   {}".format(args.experiment, scheme)
    ylabel_all = "Test accuracy ({})".format("based on all digits" if args.scenario=="class" else "after all tasks")
    ylabel = "Test accucary ({} so far)".format("based on digits" if args.scenario=="class" else "on tasks")
    x_axes = results_dict[args.methods[0]][args.seed][0]["x_task"]

    # select names / colors / ids
    names = {"none": "None", "ewc": "EWC", "si": "SI", "xdg": "XdG", "lwf": "LwF", "generative": "GR", "joint": "Joint", "bir": "BI-R"}
    colors = {"none": "grey", "ewc": "darkgreen", "si": "yellowgreen", "xdg": "deepskyblue", "lwf": "goldenrod", "generative": "red", "bir": "purple", "joint": "black"}
    names_order = ["none", "ewc", "si", "xdg", "lwf", "generative", "bir", "joint"]
    methods_ordered = sorted(args.methods, key=lambda x: names_order.index(x))

    # open pdf
    pp = plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # bar-plot
    means = [np.mean([ave_acc[seed][name] for seed in seed_list]) for name in methods_ordered]
    if args.n_seeds>1:
        sems = [np.sqrt(np.var([ave_acc[seed][name] for seed in seed_list])/(len(seed_list)-1)) for name in methods_ordered]
    figure = plt.plot_bar(means, 
                          names=[names[name] for name in methods_ordered], 
                          colors=[colors[name] for name in methods_ordered], 
                          ylabel=ylabel_all,
                          title=title,
                          yerr=sems if args.n_seeds>1 else None, 
                          ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i, name in enumerate(methods_ordered):
        if len(seed_list) > 1:
            print("{:19s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:19s} {:.2f}".format(name, 100*means[i]))
    print("#"*60)

    # line-plot
    ave_lines = []
    sem_lines = []
    for name in methods_ordered:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(acc[args.seed][name])):
            all_entries = [acc[seed][name][line_id] for seed in seed_list]
            new_ave_line.append(np.mean(all_entries))
            if args.n_seeds>1:
                new_sem_line.append(np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
        ave_lines.append(new_ave_line)
        sem_lines.append(new_sem_line)
    ylim = (0,1.02) if args.scenario=="class" else None
    figure = plt.plot_lines(ave_lines, x_axes=[2*i for i in x_axes] if args.scenario=="class" else x_axes,
                            line_names=[names[name] for name in methods_ordered], colors=[colors[name] for name in methods_ordered], title=title,
                            xlabel="# {} so far".format("digits" if args.scenario=="class" else "tasks"), ylabel=ylabel,
                            list_with_errors=sem_lines if args.n_seeds>1 else None, ylim=ylim)
    figure_list.append(figure)

    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))