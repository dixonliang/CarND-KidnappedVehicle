/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double stdv[]) {

  num_particles = 50;  // set number of particles

  default_random_engine gen; // create normal distribution

  normal_distribution<double> dist_x(x, stdv[0]);
  normal_distribution<double> dist_y(y, stdv[1]);
  normal_distribution<double> dist_theta(theta, stdv[2]);

  for (int i = 0; i < num_particles; ++i) { // set parameters for each particle

    Particle p; // create particle
    p.id = i;
    p.x = x; 
    p.y = y; 
    p.theta = theta; 
    p.weight = 1.0; // initialize weight

    p.x = p.x + dist_x(gen);
    p.y = p.y + dist_y(gen);
    p.theta = p.theta + dist_theta(gen);

    particles.push_back(p); // add particles into vector 
    
	}

  is_initialized = true; // initialize 

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  default_random_engine gen; // create normal distribution

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {

    if ( fabs(yaw_rate) < 0.000001 ) { // if yaw rate is 0 or close to 0 equations, had errors when close to 0 but not quite 0
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else { // if yaw rate is not 0 equations
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x = particles[i].x + dist_x(gen);
    particles[i].y = particles[i].y + dist_y(gen);
    particles[i].theta = particles[i].theta + dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  
  for (unsigned int i = 0; i < observations.size(); ++i) { // loop through each obseravation

    double minDistance = 100000; // intialize min dstiance for comparison to something that is very large

    int mapId = -1; // initialize map ID
    LandmarkObs obs = observations[i];

    for (unsigned j = 0; j < predicted.size(); ++j ) { // loop through each prediction for each observation
      LandmarkObs pred = predicted[j];

      if ( dist(obs.x, obs.y, pred.x, pred.y) < minDistance ) { // if found distance than previous distance , replace distance
        minDistance = dist(obs.x, obs.y, pred.x, pred.y);
        mapId = predicted[j].id;
      }
    }

    observations[i].id = mapId; // update observation to correct landmark ID based on prediciton
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  for (int i = 0; i < num_particles; ++i) { // loop through each particle

    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    vector<LandmarkObs> predictions; // create vector to hold new landmark observations
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int id = map_landmarks.landmark_list[j].id_i;

      double distance = dist(x, y, lm_x, lm_y); // find distance
      if (distance <= sensor_range) {
        predictions.push_back(LandmarkObs{id, lm_x, lm_y});
      }
    }

    vector<LandmarkObs> transform_obs; // create vector for transformed observations
    for(unsigned int j = 0; j < observations.size(); ++j) {
      double new_x = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
      double new_y = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
      transform_obs.push_back(LandmarkObs{observations[j].id, new_x, new_y});
    }

    dataAssociation(predictions, transform_obs); // use data association function match predictions with transformed observations

    particles[i].weight = 1.0; // reset weights

    for(unsigned int j = 0; j < transform_obs.size(); ++j) {
      double obs_x = transform_obs[j].x;
      double obs_y = transform_obs[j].y;

      int lm_id = transform_obs[j].id;

      double lm_x, lm_y;
      
      for (unsigned int k = 0; k < predictions.size(); ++k){
        if (predictions[k].id == lm_id){
          lm_x = predictions[k].x;
          lm_y = predictions[k].y;
        }
      }

      double weight_x = -(obs_x - lm_x); // caculate new weights and then calculate total weight using given gausian equation
      double weight_y = -(obs_y - lm_y);

      double weight = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -(weight_x*weight_x/(2*std_landmark[0]*std_landmark[0]) + (weight_y*weight_y/(2*std_landmark[1]*std_landmark[1])) ) );
      
      if (weight == 0.0) {
        particles[i].weight = 0.000001;
      } else {
        particles[i].weight *= weight;
      }
    }
  }
}

void ParticleFilter::resample() {
  default_random_engine gen;
  vector<Particle> new_particles;
  vector<double> weights;
  double max = numeric_limits<double>::min();; // set max weight 
  
  for(int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max) {
      max = particles[i].weight;
    }
  }
  
  uniform_real_distribution<double> distDouble(0.0, max); // use uniform distribution for selection of wheel
  uniform_int_distribution<int> distInt(0,num_particles - 1);

  int index = distInt(gen);

  double beta = 0.0; // sample wheel

  for(int i = 0; i < num_particles; ++i) {
    beta += distDouble(gen) * 2.0;
    while( beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}