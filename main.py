import config
import keyboard
from frametimer import FrameTimer
from predictor import Predictor
from camerasystem import CameraSystem
from config import config
from plotter import *

frame_timer = FrameTimer()

predictor = Predictor(config)
ground_truth = CameraSystem(config)

while not keyboard.is_pressed('q'):
    delta_time = frame_timer.mark()

    predicted_position = predictor.update_position(delta_time)
    real_position = ground_truth.update_position(delta_time)
    
    print(f"Predicted position: {predicted_position} Real position: {real_position}")
    
del predictor
del ground_truth

plotter.plot()
plotter.save_to_file()


# posDataL = -(np.array(posDataL) - pos - 10)
# posDataR = -(np.array(posDataR) - pos - 10)
# realPosData = -(np.array(realPosData) - pos - 10)
# mergedDopplers = [[-doppler for doppler in dopplers] for dopplers in mergedDopplers]
# usedDopplers = [-doppler for doppler in usedDopplers]

# plt.plot(timeData, posDataL, 'b')
# plt.plot(timeData, posDataR, 'lime')
# #error plt.plot(timeData, np.abs(np.array(posData) - np.array(realPosData)), 'r')
# #plt.plot(timeData, realPosData, 'r')
# plt.fill_between(timeData, realPosData - 0.5, realPosData + 0.5, facecolor='black')
# splittedDopplers = [list(dopplers) for dopplers in zip(*mergedDopplers)]
# colors = ('g', 'r', 'c', 'm', 'y')

# plt.plot(timeData, usedDopplers, 'orange')
# #mostrar una curva por doppler (desviación)
# plt.xlabel("Time (s)")
# plt.ylabel("Position (cm)")
# plt.title("Position over time")
# #plt.yticks(np.arange(, 64, 2))
# plt.grid()
# fig = plt.gcf()
# plt.show()

    # command = gui.update(tracker.get_visualization())
    
    # if command == plot:
    #     plotter.show(timeData, predictor.get_position(), tracker.get_position())
    # elif command == save:
    #     data_serializer.save(timeData, predictor.get_all_positions(), tracker.get_all_positions(), gui.ask_description())
    # elif command == exit:
    #     sys.exit(1)
    