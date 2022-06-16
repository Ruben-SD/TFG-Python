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
            positioner.update(delta_time)
            positioner.print_position()

    for positioner in positioners:
        positioner.stop()

    #plotter.print_metrics()
    # plotter.plot()
    # plotter.save_to_file()




if __name__=="__main__":
    offline = True
    if offline:
        plotting.Plotter.run_all()
    else:
        plot = True
        plotter = plotting.Plotter()
        if plot:
            plotter.load_from_file()
            plotter.plot()
        else:
            config = Config.read_config(offline=False)
            #plotter.add_data('config', config)    
            main_loop(plotter, config)