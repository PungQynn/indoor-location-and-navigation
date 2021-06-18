# Indoor Location & Navigation

This repository contains partial code to solute this [kaggle competition](https://www.kaggle.com/c/indoor-location-navigation).

My team name is "LittleChickkkkkk", click [here](https://www.kaggle.com/c/indoor-location-navigation/leaderboard) to check.

## My approach to this competition:
1. Waypoint pair

I found all areas where people are able to walk are covered by artificial polygons, and these polygons consist of several waypoint pairs. Now I can make a list including all possible waypoint pairs for individual map, those absent waypoint pairs have more possiblity present in test. 

2. Sensor data 

Sensor data was clearly the key information to reconstruct the trajectories in this competition. When using sensor data, I focused on predicting distance from one waypoint to the next one. The model constructed by few Conv1d layers with different kenerl.

3. WIFI data

I considered KNN, LightGBM and NN to deal with WIFI data, but only KNN was used in final solution. I generated lots of WIFI datas in the points within waypoints by first-order Taylor approximation.

4. Waypoint generation

There is a lot more structure to the waypoints than merely filling the empty space in corridors. I attempted to fill plausible waypoint locations

5. Time leaks

The device leak was very useful to identify the floor of test path, and narrow down candidate start waypoints range. 



