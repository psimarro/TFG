import pandas
import matplotlib.pyplot as plt
import glob
import os

PLOT_NAME = "E_FourPeople"
TRAINED_POLICY = "prueba2_E_FourPeople"

baselines = glob.glob("csv/baselines/{}*".format(PLOT_NAME))

trained = glob.glob("csv/{}*".format(TRAINED_POLICY))

fields = []
baselines_dict = {x : {} for x in baselines}
trained_dict = {x: {} for x in trained}

for result in baselines_dict:
    df = pandas.read_csv(result)
    baselines_dict[result]["step"] = df.step
    baselines_dict[result]["fps"] = df.fps
    baselines_dict[result]["reward"] = df.reward
    baselines_dict[result]["action"] = df.action

for result in trained_dict:
    df = pandas.read_csv(result)
    trained_dict[result]["step"] = df.step
    trained_dict[result]["fps"] = df.fps
    trained_dict[result]["reward"] = df.reward
    trained_dict[result]["action"] = df.action

fig, (fps, rewards, actions) = plt.subplots(1,3)
array = []
names = []
fps.set(xlabel='steps', ylabel='fps')
rewards.set(xlabel='steps', ylabel='reward')
actions.set(xlabel='# CPUs', ylabel='# steps')

for trained, results in baselines_dict.items():
    n_cores = os.path.splitext(trained)
    s = str.split(n_cores[0], "_")
    line_name = s[len(s)-1] if s[len(s)-1] == "random" else s[len(s)-2] + " " + s[len(s)-1]
    
    fps.plot(list(results["step"]), list(results["fps"]), label=line_name)
    rewards.plot(list(results["step"]), list(results["reward"]), label=line_name)
    
    array.append(list(results["action"]))
    names.append(line_name)


for trained, results in trained_dict.items():
    line_name = os.path.basename(trained)
    
    #line_name = s[len(s)-1] if s[len(s)-1] == "random" else s[len(s)-2] + " " + s[len(s)-1]
    
    fps.plot(list(results["step"]), list(results["fps"]), label=line_name)
    rewards.plot(list(results["step"]), list(results["reward"]), label=line_name)
    
    array.append(list(results["action"]))
    names.append(line_name)
    
actions.hist(array, bins=[1,2,3,4,5,6,7], label=names, histtype='bar')

plt.legend()
plt.show()

