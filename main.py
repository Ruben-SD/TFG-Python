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
        delta_time = 0.04063
        
        for positioner in positioners:
            positioner.print_position()
            positioner.update(delta_time)

    for positioner in positioners:
        positioner.print_position()
        positioner.stop()

if __name__=="__main__":
    offline = False
    if offline: # Run all files on folder in offline mode
        plotting.Plotter.run_saved(filename='23-06-2022_17-55-25.json')
    else:
        plot = False
        plotter = plotting.Plotter()
        if plot: # Simply plot one file
            plotter.load_from_file()
            plotter.print_metrics()
            plotter.plot()
        else:
            config = Config.read_config()#bestis17-06-2022_17-50-18.json#filename='17-06-2022_13-13-15.json', offline=True) # Run positioning in real time (online) mode11
            plotter.add_data('config', config)    
            main_loop(plotter, config)
            plotter.print_metrics()
            plotter.plot()
            plotter.save_to_file()