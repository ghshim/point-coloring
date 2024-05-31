import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cal_fps(inference_times):
    mean_inference_times = np.mean(inference_times)    
    return np.round(1/mean_inference_times, 2)

def draw_plot(x, y, title, xlabel, ylabel, xlim=None, ylim=None, save=False, save_path=None):
    plt.figure(figsize=(15, 7))
    plt.clf() # initialize figure 

    sns.set_style('whitegrid')

    sns.lineplot(x=x, y=y, legend=False)

    # set xlim and ylim
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)

    # draw xlabel, ylabel, and title
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    
    if save:
        # save figure
        plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal_fps', action='store_true')
    parser.add_argument('--draw_loss', action='store_true')
    parser.add_argument('--save', action='store_true')
    
    parser.add_argument('--fps_path', type=str, default='../results/fps.json', help='the directory of data')
    parser.add_argument('--train_loss_path', type=str, default='../results/train_loss.json', help='the directory of data')
    parser.add_argument('--val_loss_path', type=str, default='../results/val_loss.json', help='the directory of data')
    
    args = parser.parse_args()
    
    if args.cal_fps:
        with open(args.fps_path, "r") as fps_file:
           inference_times = json.load(fps_file)
        fps = cal_fps(list(inference_times.values()))
        print(f"FPS: {fps}")
    
    if args.draw_loss:
        with open(args.train_loss_path, "r") as train_loss_file:
            train_loss = json.load(train_loss_file)
        with open(args.val_loss_path, "r") as val_loss_file:
            val_loss = json.load(val_loss_file)
        # print(train_loss)
        # print(val_loss)
        
        draw_plot(list(range(1, 301)), train_loss, title="Train Loss", xlabel="Epoch", ylabel="Loss", 
                  save=True, save_path='../results/fig/train_loss.png')
        draw_plot(list(range(1, 301, 2)), val_loss, title="Validation Loss", xlabel="Epoch", ylabel="Loss", 
                  save=True, save_path='../results/fig/val_loss.png')
        
    print("Done.")