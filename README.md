# WaterBottleFlipProject2

This is a new and improved version of my water bottle angular velocity tracking project, now featuring different volume of water! 
This is actually for our "end of the year" science project.


## Results

Here's some graphs of angular velocity vs. amounts of water:


![bunch of graphs](https://github.com/lishangqiu/WaterBottleFlipProject2/blob/main/graphs_combined.png)


## How it works

This basically uses the two red strips on the water bottle. 
The "algorithm" finds all the red in the frame of a slow-mo video(which includes my hand), 
find the lowest two contours at the first frame, and tracks it for the rest of the video. 
It then uses the center of the contours as the two end points of a line, and we take the angle of that line to calculate the angular velocity.

Below is a demonstration of it:
![Video of the "algorithm' in working](https://github.com/lishangqiu/WaterBottleFlipProject2/blob/main/process_vid.gif)
