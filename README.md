# Checking if a parking slot is available with OpenCV

The idea is to automate the process of counting how many free spaces are left, and where to find, that is, guiding the driver to a slot that can be used.

![](https://github.com/sudo-Erno/Parking-Counter/blob/master/media/Parking.gif)

Files:
* mark_slots.py: Used for marking the parking slots and save the coordinates on the desire file to then draw them on the main code.
* testing_parking_slots.py: Used for testing if the code is working fine.
* video_reader.py: File used to run the video and return each frame.
* slots_coordinates.txt: Text file containing all the cooridantes of the parking slots. The order is as follows: ID,xi,yi,xf,yf


TODO:
* [x] Improve the text showing the total availables slots.
* [ ] Write in the slot box the slot ID.
* [x] Add the flags so it can be run via terminal.
    * [x] --video // Load the path of the desire video.
    * [x] --slots // Load the path of the file containing all the coordinates of the parking slots.
    * [x] --source // Load the corresponding camera to capture the video.
* [ ] Use a pretrained model to detect if there is any car at a parking slot.


Dependencies:
* OpenCV 4.5.4.58
* Python 3.8.0


Resources:
* https://github.com/codegiovanni/Parking_space_counter
* https://www.youtube.com/watch?v=caKnQlCMIYI
