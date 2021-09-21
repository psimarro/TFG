import pandas
import matplotlib.pyplot as plt
import glob
import os

VIDEOS =     ["E_FourPeople",
              "Mobcal",
              "QuarterBackSneak1",
              "SlideShow",
              "OldTownCross",
              "SVT04a",
              "BT709Parakeets"]

PLOT_NAME = VIDEOS[6]

NUM_STEPS = "130k"
BASELINES_PATH='csv/{}/baselines/{}*'
CSV_PATH = 'csv/{}/*{}*'

ALPHA_BASE = 1
ALPHA_T = 0.3
LEGEND_LOC = 5

BASELINE_CHOSEN = '4 cores'

baselines = glob.glob(BASELINES_PATH.format(PLOT_NAME,PLOT_NAME))

trained = glob.glob(CSV_PATH.format(PLOT_NAME, PLOT_NAME))

baselines_dict = {x : {} for x in baselines}
trained_dict = {x: {} for x in trained}

for result in baselines_dict:
    df = pandas.read_csv(result)
    baselines_dict[result]["step"] = df.step
    baselines_dict[result]["fps"] = df.fps
    baselines_dict[result]["reward"] = df.reward.cumsum()
    baselines_dict[result]["action"] = df.action
    baselines_dict[result]["markers"] = list(df.index[(df.fps >= 20) & (df.fps <= 30)])
    

for result in trained_dict:
    df = pandas.read_csv(result)
    
    trained_dict[result]["step"] = df.step
    trained_dict[result]["fps"] = df.fps
    trained_dict[result]["reward"] = df.reward.cumsum()
    trained_dict[result]["action"] = df.action
    trained_dict[result]["markers"] = list(df.index[(df.fps >= 20) & (df.fps <= 30)])

fig1, (fps, rewards) = plt.subplots(1,2)
fig1.suptitle(PLOT_NAME)

fps.set(xlabel='pasos', ylabel='fps')
rewards.set(xlabel='pasos', ylabel='recomponsa acumulada')

actions = []
line_names = []

for trained, results in baselines_dict.items():
    n_cores = os.path.splitext(os.path.basename(trained))
    s = str.split(n_cores[0], "_")
    line_name = s[len(s)-1] if s[len(s)-1] == "random" else s[len(s)-2] + " " + s[len(s)-1]
    
    if line_name == BASELINE_CHOSEN or BASELINE_CHOSEN == 'all':
        fps.plot(list(results["step"]), list(results["fps"]), label=line_name, alpha=ALPHA_BASE)
        rewards.plot(list(results["step"]), list(results["reward"]), label="_" , alpha=ALPHA_BASE)
        rewards.annotate(str(list(results["reward"])[49]), 
                    (50, list(results["reward"])[49]), 
                    textcoords="offset points",
                    xytext=(0,0), 
                    ha='center', alpha=ALPHA_BASE) 
    else: 
        pass

    actions.append(list(results["action"]))
    line_names.append(line_name)


for trained, results in trained_dict.items():
    line_name = "entrenamiento " + NUM_STEPS + " pasos "
    
    fps.plot(list(results["step"]), list(results["fps"]), label=line_name, color='blue', alpha=ALPHA_T)
    fps.plot([x+1 for x in results["markers"]],[list(results["fps"])[i] for i in results["markers"]], ls="", marker="o", label="_", alpha=ALPHA_T)
    rewards.plot(list(results["step"]), list(results["reward"]), label="_", color='blue',alpha=ALPHA_T)
    rewards.annotate(str(list(results["reward"])[49]), # this is the text
                 (50, list(results["reward"])[49]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,0), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    actions.append(list(results["action"]))
    line_names.append(line_name)


fps.hlines(y=20, xmin=0, xmax=50,linestyle='dotted', color='black', label='objetivo')
fps.hlines(y=30, xmin=0, xmax=50,linestyle='dotted', color='black')
fig1.legend(loc=LEGEND_LOC)

fig2, (h) = plt.subplots(1,1)
n, bins, patches = h.hist(actions, bins=range(1,10), label=line_names, histtype='bar')
#print labels for each rectangle
for patch, data in zip(patches, n):
    for rect, label in zip(patch, data):
        if label != 0.0:
            height = rect.get_height()
            h.text(rect.get_x() + rect.get_width() / 2, height+0.01, int(label),
                    ha='center', va='bottom')
h.set(xlabel="nº CPUs", ylabel="nº pasos")
h.set_title("Uso de CPU " + PLOT_NAME)
h.legend()

plt.show()

