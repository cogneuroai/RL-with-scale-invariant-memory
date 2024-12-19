import wandb
import pickle
import os
import json

def collect_performances(entity, project, run_id_dict, f_name):
    '''

    :param entity:
    :param project:
    :param run_id_dict:
    :param f_name:
    :return:
    '''

    performances = {}
    # go through the run_id_dict, collect the performances in a 2d matrix for each network type
    for key in run_id_dict.keys():
        performances[key] = {'values': [], 'global_steps': []}
        print('Collecting performances for {}.'.format(key))
        for run_id in run_id_dict[key]:
            print('Run id: {}'.format(run_id))
            run = api.run(entity + '/' + project + '/' + run_id)
            performance_history = run.scan_history(keys=["performance"], page_size=50)
            step_history = run.scan_history(keys=["steps/global"], page_size=50)
            performance_run = [row["performance"] for row in performance_history]
            global_steps = [row["steps/global"] for row in step_history]

            # print(performances, global_steps)
            performances[key]['global_steps'].append(global_steps)
            performances[key]['values'].append(performance_run)

    # save the performance dictionary in a pickle
    with open("postprocessing/data/" + f_name + ".pkl", 'wb') as fp:
        pickle.dump(performances, fp)


if __name__ == '__main__':
    os.makedirs("postprocessing/data/", exist_ok=True)
    api = wandb.Api()
    with open("configs_performance.json", 'r') as json_file:
        parsed_json = json_file.read()
    config_json = json.loads(parsed_json)
    entity = config_json['entity']
    project = config_json['project']
    run_ids = config_json['run_ids']
    f_name = 'performances_2D_dt_100'
    collect_performances(entity, project, run_ids, f_name)
