

import numpy as np
import matplotlib.pyplot as plt
import os,shutil
import imageio
import imageio_ffmpeg


# This is an agent-based simulation of disease spread hacked from the MATLAB script of [Joshua Gafford](https://www.mathworks.com/matlabcentral/fileexchange/74610-simulitis-a-coronavirus-simulation) who was making a model inspired by the COVID-19 simulations posted by [Washington Post](https://www.washingtonpost.com/graphics/2020/world/corona-simulator/). This is just a qualitative illustration of the effects of isolating portions of the public (oh, hi there "social distancing") to reduce the number of encounters and thus transmission. 
# 
# Packages can all be installed via conda or pip save for `imageio-ffmpeg` which needs to come from `pip`. 
# 
# Output is a gif and movie file (each) with an animation of the simulated outbreak and resulting curve of infections over time. The filename includes the isolation percentage.

# define some variables
#       lim:  spatial limits [scalar > 0]
#       n:  number of carriers [scalar > 0]
#       rad:  infection radius in distance units [scalar > 0]
#       speed:  initial carrier speed in units/day [scalar > 0]
#       iso: percentage of social isolation [0 <= scalar < 1]
#       t_tot: total simulation time in days [scalar > 0]
#       delT: simulation time increment in days [scalar > 0]
#       t_recovery: mean recovery time in days [scalar > 0]
#       p_init: probability of carriers initially infected [0 <= scalar < 1]
#       p_trans: probability of disease transmission [0 <= scalar < 1]
#       p_mort: mortality rate [0 <= scalar < 1]

class agent(Object):

	def __init__(self, lim=200, n_delay = 10, iso = 0.5, rad = 2.5, speed = 10, t_recovery = 14,
	p_trans = 0.99, p_mort =0.02, p_init = 0.015, t_tot = 70, delT = 0.5, agent_idx = 0):
		# read in parameters
		self.rad = rad
		self.speed = speed
		self.t_recovery = t_recovery
		self.p_trans = p_trans
		self.p_mort = p_mort
		self.lim = lim
		self.iso = iso
		self.p_init = p_init
		self.t_tot  t_tot
		self.delT = delT
		self.ndelay = n_delay
		self.agent_idx = agent_idx # index of which agent you are in the list of all agents - for collisions

		# initialize some states for location
		self.isolate = np.random.random() < self.iso
		self.pos = np.zeros((self.t_tot * self.delT,2))
		self.th = np.random.random()*2*np.pi
		if not self.isolate:
			self.v = [self.speed * np.cos(th), self.speed * np.sin(th)]
			self.pos[0, :] = np.random.random(2) * self.lim
		else:
			self.v = [0,0]
		# initialize some states for health
		self.infected = np.random.random() < self.p_init
		self.healthy = ~self.infected
		self.recovered = 0
		self.dead = 0
		self.p_death = np.random.random() < self.p_mort
		self.t_rec = np.ceil(self.t_recovery  + np.random.normal(loc=0, scale=4, size=1)) / self.delT

	def move(self, iteration, th=None, v=None):
		# first move ....

		# option to either provide th or v
		# if no v was provided, check for theta
		if self.v is not [0,0]:
			if self.v is None:
				# if also no theta provided, assume velocity is unchanged from last time step and carry on
				if th is not None:
					self.v = [self.speed * np.cos(th), self.speed * np.sin(th)]
		self.pos[iteration,:] = self.pos[iteration-1,:] + v

		# next health increment ...
		# first, if infected and recovery time is up, either recover of decease
		if self.infected and self.t_rec <= 0:
			self.infected = 0
			self.healthy = 0
			if self.p_death:
				self.v=[0,0]
				self.dead = 1
				self.recovered = 0
			else:
				self.dead = 0
				self.recovered = 1

		elif self.infected:
			# one day closer to recovery
			self.t_rec -= 1

		def interact(self, allpos):
			# now interact with all the others

class population(Object):
	def __init__(self, n=200, lim=200, n_delay = 10, iso = 0.5, rad = 2.5, speed = 10, t_recovery = 14,
	p_trans = 0.99, p_mort =0.02, p_init = 0.015, t_tot = 70, delT = 0.5):
		self.n = n
		self.lim = lim
		self.n_delay = n_delay
		self.t_tot = t_tot
		self.delT  = delT
		# initially just pass through all the parameters. Could randomize some here
		self.peeps = [agent(lim, n_delay, iso, rad, speed, t_recovery,
	    				p_trans, p_mort, p_init, t_tot, delT) for i in n]

	def increment_time(self, citer):

		# move all
		for i,cp in enumerate(self.peeps):
			cp.move(citer, th[i])
		# interact

		# report out




pop = population(n=200, lim=200, n_delay = 10, iso = 0.5, rad = 2.5, speed = 10, t_recovery = 14,
	p_trans = 0.99, p_mort =0.02, p_init = 0.015, t_tot = 70, delT = 0.5)


for citer in range(np.int(self.t_tot/self.delT)):
	pop.increment_time(citer)

'''
def run_simulation(n = 200, , make_gif=True, make_mp4=True):

	

	if os.path.exists('figures'):
	    shutil.rmtree('figures')
	os.mkdir('figures')

	# set up solution vectors
	inf_sum = np.zeros(np.int(t_tot/delT))      # Percentage of infected carriers
	hea_sum = inf_sum.copy()                      # Percentage of unaffected carriers
	rec_sum = inf_sum.copy()                      # Percentage of recovered carriers
	dead_sum = inf_sum.copy()                     # Percentage of dead carriers
	cumulative_sum = inf_sum.copy()               # Percentage of cumulative disease cases





	for i in range(np.int(t_tot/delT)):
	    print('Timestep {0} out of {1}: iso={2:.3f}'.format((i+1), np.int(t_tot/delT),iso))
	    # Decrement collision delay
	    collision = collision-np.ones((n,n))
	    collision[collision<0]=0
	    
	    # Update carrier position
	    pos_new = pos + v*(~isolate)*delT
	    
	    # Step through each carrier
	    for k in range(n):
	        
	        # If recovery time is up, carrier is either recovered or dead
	        if infected[k] and t_rec[k]<=0:
	            
	            # If recovery time is up and carrier is dead, well, it's dead.
	            # Zero it's velocity
	            if p_death[k]:
	                dead[k] = 1;
	                v[k,:] = [0, 0]
	                recovered[k]=0
	                infected[k]=0
	                healthy[k]=0
	            else:
	                # If recovery time is up and carrier is not dead, recover
	                # that carrier
	                recovered[k]=1
	                infected[k]=0
	                healthy[k]=0
	                dead[k]=0

	            
	        # If carrier is infected and not recovered, decrement recovery time
	        elif infected[k]:
	            t_rec[k] -= 1

	        
	        # Step through all other carriers, looking for collisions, and if
	        # so, transmit disease and recalculate trajectory
	        for j in range(n):
	            
	            if j != k:
	                
	                # Get positions of carriers j and k
	                pos0 = pos_new[k,:]
	                pos1 = pos_new[j,:]
	                
	                # If collision between two living specimens, re-calcuate
	                # direction and transmit virus (but don't check the same
	                # two carriers twice)
	                if (np.linalg.norm(pos0-pos1)<=(2*rad)) and (1 - collision[k,j]) and (1 - collision[j,k]):
	                     
	                    # Create collision delay (i.e. if carrier j and k have
	                    # recently collided, don't recompute collisions for a
	                    # n_delay time steps in case they're still close in proximity, 
	                    # otherwise they might just keep orbiting eachother)
	                    collision[k,j] = n_delay
	                    collision[j,k] = n_delay
	                    
	                    # Compute New Velocities
	                    phi = np.arctan2((pos1[1]-pos0[1]),(pos1[0]-pos0[0]))
	                    
	                    # if one carrier is isolated, treat it like a wall and
	                    # bounce the other carrier off it
	                    if isolate[j] or dead[j]:
	                        
	                        # Get normal direction vector of 'virtual wall'
	                        phi_wall = -phi+np.pi/2;
	                        n_wall = [np.sin(phi_wall), np.cos(phi_wall)];
	                        dot = v[k,:].dot(np.array(n_wall).T)
	                        
	                        # Redirect non-isolated carrier
	                        v[k,0] = v[k,0]-2*dot*n_wall[0]
	                        v[k,1] = v[k,1]-2*dot*n_wall[1]
	                        v[j,0] = 0
	                        v[j,1] = 0
	                        
	                    elif isolate[k] or dead[k]:
	                        
	                        # Get normal direction vector of 'virtual wall'
	                        phi_wall = -phi+np.pi/2
	                        n_wall = [np.sin(phi_wall), np.cos(phi_wall)]
	                        dot = v[j,:].dot(np.array(n_wall).T)
	                        
	                        # Redirect non-isolated carrier
	                        v[j,0] = v[j,0]-2*dot*n_wall[0]
	                        v[j,1] = v[j,1]-2*dot*n_wall[1]
	                        v[k,0] = 0
	                        v[k,1] = 0
	                        
	                    # Otherwise, transfer momentum between carriers
	                    else:                        
	                        # Get velocity magnitudes
	                        v0_mag = np.sqrt(v[k,0]**2+v[k,1]**2)
	                        v1_mag = np.sqrt(v[j,0]**2+v[j,1]**2)
	                        
	                        # Get directions
	                        th0 = np.arctan2(v[k,1],v[k,0]);
	                        th1 = np.arctan2(v[j,1],v[j,0]);
	                        
	                        # Compute new velocities
	                        v[k,0] = v1_mag*np.cos(th1-phi)*np.cos(phi)+v0_mag*np.sin(th0-phi)*np.cos(phi+np.pi/2)
	                        v[k,1] = v1_mag*np.cos(th1-phi)*np.sin(phi)+v0_mag*np.sin(th0-phi)*np.sin(phi+np.pi/2)
	                        v[j,0] = v0_mag*np.cos(th0-phi)*np.cos(phi)+v1_mag*np.sin(th1-phi)*np.cos(phi+np.pi/2)
	                        v[j,1] = v0_mag*np.cos(th0-phi)*np.sin(phi)+v1_mag*np.sin(th1-phi)*np.sin(phi+np.pi/2)
	                                            
	                    # If either is infected and not dead...
	                    if (infected[j] or infected[k]) and ((1-dead[k]) or (1-dead[j])):
	                        
	                        # If either is recovered, no transmission
	                        if recovered[k]:
	                            infected[k]=0
	                        elif recovered[k]:
	                            infected[j]=0
	                            
	                        # Otherwise, transmit virus
	                        else:
	                            transmission = np.random.random(1)[0]<p_trans
	                            if transmission:
	                                infected[j]=1
	                                infected[k]=1
	                                healthy[j]=0
	                                healthy[k]=0
	        # Look for collisions with walls and re-direct
	        
	            # Left Wall
	            if pos_new[k,0]<=rad:
	                if v[k,0]<0:
	                    v[k,0]=-v[k,0]            
	            # Right wall
	            elif pos_new[k,0]>=lim-rad:
	                if v[k,0]>0:
	                    v[k,0]=-v[k,0]
	            
	            # Bottom Wall
	            if pos_new[k,1] <=rad:
	                if v[k,1]<0:
	                    v[k,1]=-v[k,1]

	            # Top Wall
	            elif pos_new[k,1] >=(lim-rad):
	                if v[k,1]>0:
	                    v[k,1]=-v[k,1]
	                    
	    # Update color vector
	    color=np.hstack([infected, healthy, recovered]*(1-dead))    
	    color = np.hstack((color,dead))
	    # Update solution vectors
	    inf_sum[i] = sum(infected)*100/n
	    hea_sum[i] = sum(healthy)*100/n
	    rec_sum[i] = sum(recovered)*100/n
	    dead_sum[i] = sum(dead)*100/n
	    cumulative_sum[i] = 100-hea_sum[i]
	    
	    
	    # make plots
	    # first plot positions
	    fig, ax = plt.subplots(2,1,figsize=(5,10))
	    

	    infidx = np.where(color[:,0]==1)[0]
	    ax[0].scatter(pos_new[infidx,0],pos_new[infidx,1], c='red')
	    healthyidx = np.where(color[:,1]==1)[0]
	    ax[0].scatter(pos_new[healthyidx,0],pos_new[healthyidx,1], c='blue')
	    deadidx = np.where(color[:,3]==1)[0]
	    ax[0].scatter(pos_new[deadidx,0],pos_new[deadidx,1], c='black')

	    recovidx = np.where(color[:,2]==1)[0]
	    ax[0].scatter(pos_new[recovidx,0],pos_new[recovidx,1], c='blue', alpha=.5)
	    ax[0].axis('square')    
	    ax[0].set_xticks([])
	    ax[0].set_yticks([])
	    ax[0].set_xlim([0,lim])
	    ax[0].set_ylim([0,lim])
	    
	    ax[0].set_title('where my peeps?')
	    plt.suptitle('Isolation = {}%\n'.format(iso*100) + 'Time = {:.2f} days'.format(i*delT))
	    
	    # now plot the distributions evolving over time
	    
	    ax[1].fill_between(t[:i],(inf_sum[:i]+dead_sum[:i]+hea_sum[:i]),
	                       (inf_sum[:i]+dead_sum[:i]+hea_sum[:i]+rec_sum[:i]),color='blue', alpha=.5)
	    ax[1].fill_between(t[:i],(inf_sum[:i]+dead_sum[:i]),
	                       (inf_sum[:i]+dead_sum[:i]+hea_sum[:i]),color='blue')
	    ax[1].fill_between(t[:i],dead_sum[:i],
	                       (inf_sum[:i]+ dead_sum[:i]),color='r')
	    ax[1].fill_between(t[:i],np.zeros(i),dead_sum[:i], color='k')
	    
	    
	    ax[1].set_xlim([0,t_tot])
	    ax[1].set_ylim([0,100])
	    ax[1].set_title('How many infected?')
	    ax[1].set_xlabel('Days')
	    ax[1].set_label('Percent of population')
	    plt.legend(['recovered', 'healthy','infected','dead'])
	    plt.savefig('figures/fig{0}.png'.format(i))
	    
	    #print(inf_sum[i]+dead_sum[i]+hea_sum[i]+rec_sum[i])
	    plt.close('all')
	    
	    pos=pos_new


	if make_gif:
		frames_path = 'figures/fig{i}.png'
		with imageio.get_writer('model{}.gif'.format(iso), mode='I') as writer:
		    for i in range(np.int(t_tot/delT)):
		        writer.append_data(imageio.imread(frames_path.format(i=i)))

	if make_mp4:
		frames_path = 'figures/fig{i}.png'
		with imageio.get_writer('model{}.mp4'.format(iso), mode='I') as writer:
		    for i in range(np.int(t_tot/delT)):
		        writer.append_data(imageio.imread(frames_path.format(i=i)))



if __name__ == '__main__':
	# sweep a few isolation values
	for ciso in [0,  .5,  .75]:
		run_simulation(iso=ciso, t_tot=90)
'''