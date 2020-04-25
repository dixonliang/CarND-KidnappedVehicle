# Kidnapped Vehicle Project

This was the sixth project in Udacity's Self Driving Car NanoDegree. The task was to create a particle filter to localize a simulated vehicle. In the src files "particle_filter.cpp", you will find my code where I implemented serveral functions. 

I began with initializing the filter using initial given GPS position estimates with 50 particles. Then, I implemented a prediction fuction using a motion model. Using new sensor information, an update function then associates readings to map landmarks to update the weights of the particules. A uniform distribution was also used to resample the particles after weights had been updated. 
