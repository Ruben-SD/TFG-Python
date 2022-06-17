import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import config
import keyboard
from frametimer import FrameTimer
from positionerfactory import PositionerFactory
from config import Config
import plotting

def main_loop(plotter, config):
    positioners = PositionerFactory.create_positioners(config, plotter)

    frame_timer = FrameTimer(config, plotter)
    while not keyboard.is_pressed('esc') and not frame_timer.reached_end():
        delta_time = frame_timer.mark()
        #print(1/delta_time)
        for positioner in positioners:
            positioner.print_position()
            positioner.update(delta_time)

    for positioner in positioners:
        positioner.print_position()
        positioner.stop()

    #plotter.print_metrics()
    # plotter.plot()
    # plotter.save_to_file()




if __name__=="__main__":
    offline = False
    if offline: # Run all files on folder in offline mode
        plotting.Plotter.run_saved('16-06-2022_15-10-03.json')
    else:
        plot = False
        plotter = plotting.Plotter()
        if plot: # Simply plot one file
            plotter.load_from_file()
            plotter.print_metrics()
            plotter.plot()
        else:
            config = Config.read_config(offline=True) # Run positioning in real time (online) mode
            plotter.add_data('config', config)    
            main_loop(plotter, config)
            plotter.print_metrics()
            plotter.plot()