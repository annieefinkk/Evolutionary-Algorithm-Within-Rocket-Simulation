'''
This program simulates a population of rockets trying to reach a target 
while navigating through a field of planets with gravity using an evolutionary algorithm. 
Each rocket has a genome that with three traits: initial velocity, mass, and thrust. The rockets 
evolve over generations using tournament selection, crossover, and mutation.

The simulation visualizes the rockets' trajectories and the planets, and prints out statistics 
about each generation's performance. There is a physical visualiztion that will automatically display 
when this program is run. 

Simply run this program like running any python file. There are no extra arguments, and the parameters can 
be changed in the code below. Also, the three different graph configurations used in my report can be 
switched between by commenting/uncommenting the relevant sections below.

'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
plt.style.use('dark_background')



# Simulation parameters
G = 22 # Gravitational constant (how much the planet gravity will pull on the rockets)
dt = 0.05 # time step for simulation updates
num_frames = 5000 # how long each generation has to reach target before evolving population
num_generations = 50 # how many generations to evolve
num_rockets = 100 # how many rockets in each generation
p_m = 0.07 # mutation probability
p_c = 0.3 # crossover probability
ts = 4 # tournament size
TARGET_REACHED_DIST = 10.0 # target radius to be considered a successful hit



#=========================================================================================================
#   GRAPH 1: DENSE

# win_x, win_y = (-200, 800), (-200, 800)
# target = {'pos': np.array([620.0, 640.0])} 
# start_pos = np.array([0.0, -100.0])
# START_TO_TARGET_DIST = np.linalg.norm(target['pos'] - start_pos)

# planets = [
#     {'pos': np.array([50, 50]),    'mass': 6000, 'radius': 5},
#     {'pos': np.array([100, -100]), 'mass': 6000, 'radius': 5},
#     {'pos': np.array([-50, -50]),  'mass': 6000, 'radius': 5},
#     {'pos': np.array([50, -175]),  'mass': 6000, 'radius': 5},
#     {'pos': np.array([100, 0]),    'mass': 12000, 'radius': 15}, 
#     {'pos': np.array([100, 100]),  'mass': 12000, 'radius': 6}, 
#     {'pos': np.array([100, 200]),  'mass': 12000, 'radius': 25}, 
#     {'pos': np.array([-150, -150]),'mass': 12000, 'radius': 40}, 
#     {'pos': np.array([300, 300]),  'mass': 9000,  'radius': 50}, 
#     {'pos': np.array([600, 400]),  'mass': 7000,  'radius': 60}, 
#     {'pos': np.array([100, 500]),  'mass': 10000, 'radius': 25}, 
#     {'pos': np.array([0, 400]),    'mass': 15000, 'radius': 20}, 
#     {'pos': np.array([750, 750]),  'mass': 20000, 'radius': 15},
#     {'pos': np.array([500, 100]),  'mass': 15000, 'radius': 15},
#     {'pos': np.array([200, -50]),  'mass': 5000,  'radius': 20},
#     {'pos': np.array([400, 700]),  'mass': 11000, 'radius': 15},
#     {'pos': np.array([-50, 750]),  'mass': 9500,  'radius': 18},
#     {'pos': np.array([-180, 200]), 'mass': 4000,  'radius': 8},
#     {'pos': np.array([-100, 50]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([-180, 200]), 'mass': 4000,  'radius': 8},
#     {'pos': np.array([-100, 50]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([-20, 600]),  'mass': 4000,  'radius': 30},
#     {'pos': np.array([50, 120]),   'mass': 4000,  'radius': 8},
#     {'pos': np.array([150, 250]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([180, 480]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([220, 150]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([260, 550]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([320, 50]),   'mass': 4000,  'radius': 8},
#     {'pos': np.array([350, 420]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([380, -150]), 'mass': 4000,  'radius': 40},
#     {'pos': np.array([420, 300]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([450, 780]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([480, 520]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([520, -20]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([550, 650]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([580, 180]),  'mass': 4000,  'radius': 17},
#     {'pos': np.array([620, 720]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([650, 50]),   'mass': 4000,  'radius': 8},
#     {'pos': np.array([680, 330]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([720, 580]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([750, -100]), 'mass': 4000,  'radius': 8},
#     {'pos': np.array([780, 400]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([-120, -40]), 'mass': 4000,  'radius': 8},
#     {'pos': np.array([-60, 320]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([30, 700]),   'mass': 4000,  'radius': 8},
#     {'pos': np.array([120, -180]), 'mass': 4000,  'radius': 8},
#     {'pos': np.array([250, 680]),  'mass': 4000,  'radius': 35},
#     {'pos': np.array([390, 210]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([460, -60]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([540, 380]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([610, 80]),   'mass': 4000,  'radius': 8},
#     {'pos': np.array([670, 790]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([730, 240]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([140, 590]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([280, 740]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([410, 610]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([560, -140]), 'mass': 4000,  'radius': 100},
#     {'pos': np.array([690, 460]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([760, 670]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([-40, 180]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([170, -30]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([490, 410]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([640, 560]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([710, 10]),   'mass': 4000,  'radius': 8},
#     {'pos': np.array([-140, 510]), 'mass': 4000,  'radius': 8},
#     {'pos': np.array([210, 760]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([370, -90]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([530, 220]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([600, 600]),  'mass': 4000,  'radius': 8},
#     {'pos': np.array([770, 310]),  'mass': 4000,  'radius': 8},
# ]
#=========================================================================================================


#=========================================================================================================
#   GRAPH 2 - SYMMETRY

# win_x, win_y = (-500, 500), (-500, 500)
# start_pos = np.array([0.0, -200.0])
# target = {'pos': np.array([0.0, 400.0])}
# START_TO_TARGET_DIST = np.linalg.norm(target['pos'] - start_pos)

# m = 2000 # Uniform moderate mass
# r = 60    # Uniform moderate radius

# planets = [
#     # Row 1: Immediate choices (Staggered)
#     {'pos': np.array([-150, -100]), 'mass': m, 'radius': r},
#     {'pos': np.array([150, -100]),  'mass': m, 'radius': r},

#     {'pos': np.array([-425, -100]), 'mass': m, 'radius': r},
#     {'pos': np.array([425, -100]),  'mass': m, 'radius': r},

#     {'pos': np.array([-425, 200]), 'mass': m, 'radius': r},
#     {'pos': np.array([425, 200]),  'mass': m, 'radius': r},

#     {'pos': np.array([-350, 450]),   'mass': m, 'radius': r},
#     {'pos': np.array([350, 450]),    'mass': m, 'radius': r},
    
#     # Row 2: Central blockage
#     {'pos': np.array([0, 50]),      'mass': m, 'radius': r},
#     {'pos': np.array([-300, 50]),   'mass': m, 'radius': r},
#     {'pos': np.array([300, 50]),    'mass': m, 'radius': r},
    
#     # Row 3: The Mid-Field Gate
#     {'pos': np.array([-100, 200]),  'mass': m, 'radius': r},
#     {'pos': np.array([100, 200]),   'mass': m, 'radius': r},
    
#     # Row 4: Final Target Shielding
#     {'pos': np.array([-200, 350]),  'mass': m, 'radius': r},
#     {'pos': np.array([200, 350]),   'mass': m, 'radius': r},
#     {'pos': np.array([0, 500]),     'mass': m, 'radius': r},

#     # Row 2: Central blockage
#     #{'pos': np.array([0, -250]),      'mass': m, 'radius': r},
#     {'pos': np.array([-300, -250]),   'mass': m, 'radius': r},
#     {'pos': np.array([300, -250]),    'mass': m, 'radius': r},

#     # Row 3: The Mid-Field Gate
#     {'pos': np.array([-100, -350]),  'mass': m, 'radius': r},
#     {'pos': np.array([100, -350]),   'mass': m, 'radius': r},

#     {'pos': np.array([0, -500]),      'mass': m, 'radius': r},
#     {'pos': np.array([-300, -500]),   'mass': m, 'radius': r},
#     {'pos': np.array([300, -500]),    'mass': m, 'radius': r},
# ]
#=========================================================================================================



#=========================================================================================================
# GRAPH 3: HIGH MASS 

win_x, win_y = (-400, 400), (-100, 1000)
target = {'pos': np.array([0.0, 950.0])}
start_pos = np.array([0.0, -50.0])
START_TO_TARGET_DIST = np.linalg.norm(target['pos'] - start_pos)
planets = [
    {'pos': np.array([-100, 150]), 'mass': 20000, 'radius': 5},
    {'pos': np.array([100, 150]),  'mass': 20000, 'radius': 5},
    {'pos': np.array([-35, 300]),  'mass': 18000, 'radius': 4}, 
    {'pos': np.array([35, 600]),   'mass': 18000, 'radius': 4},
    {'pos': np.array([-350, 300]), 'mass': 18000, 'radius': 6}, 
    {'pos': np.array([350, 400]),  'mass': 18000, 'radius': 6},
    {'pos': np.array([0, 450]),    'mass': 25000, 'radius': 8},
    {'pos': np.array([-150, 700]), 'mass': 22000, 'radius': 5},
    {'pos': np.array([300, 750]),  'mass': 20000, 'radius': 5},
    {'pos': np.array([60, 880]),   'mass': 18000, 'radius': 4},
    {'pos': np.array([0, 1000]),   'mass': 30000, 'radius': 10},
]
#=========================================================================================================



# fitness function
def fitness(r):
    # distance-based fitness: closer to target is higher fitness
    if r['best_dist'] < 0: return 0 # dead rockets have 0 fitness
    if r['best_dist'] < TARGET_REACHED_DIST: return 100 # winners get max fitness

    return (1 - (r['best_dist'] / START_TO_TARGET_DIST)) * 100

# calculate gravity vector from planet onto rocket
def gravity(r, p):
    r_vec = p['pos'] - r['pos']
    dist = np.linalg.norm(r_vec) + 0.00001
    return G * p['mass'] * r_vec / dist**3

# initialize population of rockets with random genomes (thrust, mass, initial velocity)
def init_population(n):
    pop = []
    for _ in range(n):
        theta = np.random.uniform(0, 2*np.pi)
        init_vel = np.array([np.cos(theta), np.sin(theta)]) * 1.5
        pop.append({
            'pos': start_pos.copy(),
            'vel': init_vel.copy(),
            'dead': False,
            'won': False,
            'genome': {
                'init_vel': init_vel.copy(), 
                'mass': np.random.uniform(1, 30), # random mass from 1 to 30
                'thrust': np.random.uniform(0, 30.0) # random thrust from 0 to 30
            },
            'trajectory': [start_pos.copy()], # for simulation visualization
            'best_dist': np.linalg.norm(start_pos - target['pos']) # track best distance
        })
    return pop

# evolve each generation using tournament selection, crossover, and mutation
def evolve(pop):
    new_pop = []

    while len(new_pop) < num_rockets:
        # choose two random parents from a pool of ts randomly selected individuals
        idx1 = np.random.choice(len(pop), ts)
        idx2 = np.random.choice(len(pop), ts)

        # choose highest fitness from each pool
        p1 = pop[max(idx1, key=lambda i: fitness(pop[i]))]
        p2 = pop[max(idx2, key=lambda i: fitness(pop[i]))]

        # extract genome
        g1, g2 = p1['genome'], p2['genome']
        g = {}

        # perform crossover based on p_c probability
        g['thrust'] = g2['thrust'] if np.random.rand() < p_c else g1['thrust']
        g['mass'] = g2['mass'] if np.random.rand() < p_c else g1['mass']
        g['init_vel'] = g2['init_vel'].copy() if np.random.rand() < p_c else g1['init_vel'].copy()

        # perform mutation on each trait based on p_m probability
        if np.random.rand() < p_m:
            g['thrust'] += np.random.normal(0, 1)

        if np.random.rand() < p_m:
            g['mass'] += np.random.normal(0, 0.5)
            g['mass'] = max(0.1, g['mass'])

        if np.random.rand() < p_m:
          theta = np.random.uniform(0, 2 * np.pi)
          g['init_vel'] = np.array([np.cos(theta), np.sin(theta)])

        # add new individual to new population
        new_pop.append({
         'pos': start_pos.copy(),
         'vel': g['init_vel'].copy(),
         'dead': False,
         'won': False,
         'genome': g,
         'trajectory': [start_pos.copy()],   
         'best_dist': np.linalg.norm(start_pos - target['pos'])
        })
    return new_pop



# =============== ANIMATION SETUP ==================
rocket_pop = init_population(num_rockets)
fig, ax = plt.subplots(figsize=(7,7))
ax.set_xlim(win_x); ax.set_ylim(win_y); ax.set_aspect('equal')

# initialize scatter with colors array
rocket_scatter = ax.scatter([], [], s=40, zorder=5)
traj_lines = [ax.plot([], [], '-', alpha=0.3, lw=1)[0] for _ in range(num_rockets)]


# add planets to map with color based on mass (heavier planets are more opaque, lighter planets are more transparent)
min_alpha = 0.3
max_alpha = 1.0

masses = [p['mass'] for p in planets]
m_min, m_max = min(masses), max(masses)

# use a colormap to assign colors based on mass and display planets
for p in planets:
    norm = (p['mass'] - m_min) / (m_max - m_min + 1e-8)
    alpha = min_alpha + norm * (max_alpha - min_alpha)
    ax.add_patch(plt.Circle(p['pos'], p['radius'], color="blue", alpha=alpha))
ax.plot(target['pos'][0], target['pos'][1], 'yo', markersize=15) # add target
ax.plot(target['pos'][0], target['pos'][1], 'ko', markersize=11) # add target

# initialize generation and frame counters
frame_counter = 0; generation = 0

# main animation loop, updates rocket positions, checks for collisions and success, evolves population when done
def update(frame):
    global rocket_pop, frame_counter, generation
    current_colors = []

    for i, r in enumerate(rocket_pop):
        # If rocket has already won or is dead, it stops moving
        if r['dead'] or r['won']:
            current_colors.append('gold' if r['won'] else 'red')
            continue

        # calculate gravitational acceleration from all planets
        # if rocket is too close to a planet, it dies and is removed from simulation
        acc = np.zeros(2)
        for p in planets:
            p_dist = np.linalg.norm(r['pos'] - p['pos'])

            # if rocket collides with planet
            if p_dist < p['radius']:
                r['dead'] = True
                r['vel'] = np.array([0.0, 0.0])
                d = np.linalg.norm(r['pos'] - target['pos'])
                if d < r['best_dist']: r['best_dist'] = d 
                break
            # otherwise, apply gravity from planet
            acc += gravity(r, p)
        

        # if rocket is still alive, apply thrust and update position
        if not r['dead'] and not r['won']:
            vel_mag = np.linalg.norm(r['vel'])
            
            # apply thrust in direction of current velocity, scaled by thrust/mass and time step
            acc += (r['genome']['thrust'] / r['genome']['mass']) * (r['vel'] / vel_mag)

            # update velocity and position
            r['vel'] = (r['vel'] + acc * dt)
            r['pos'] += r['vel'] * dt
            
            # update best distance to target
            d = np.linalg.norm(r['pos'] - target['pos'])
            if d < r['best_dist']: r['best_dist'] = d
            
            # check if rocket has reached target
            if d < TARGET_REACHED_DIST:
                r['won'] = True
                r['vel'] = np.array([0.0, 0.0])

            # out of bounds check, rocket dies to it flies too far away from the window
            if ((r['pos'][0] < (win_x[0] - 300)) or (r['pos'][0] > (win_x[1] + 300))) or (r['pos'][1] < (win_y[0] - 300) or r['pos'][1] > (win_y[1] + 300)):
                r['dead'] = True
                r['vel'] = np.array([0.0, 0.0])
                r['best_dist'] = -1
                traj_lines[i].set_data([], [])
                continue

            # draws trajectory of rockets (for simulation visuals)
            r['trajectory'].append(r['pos'].copy())
            recent = np.array(r['trajectory'][-80:])
            traj_lines[i].set_data(recent[:,0], recent[:,1])
            current_colors.append('red')
        else:
            traj_lines[i].set_data([], [])

    # update scatter positions and colors
    # offset dead rockets to hide them, keep winners on target
    offsets = []
    for r in rocket_pop:
        if r['dead']: offsets.append([999, 999])
        else: offsets.append(r['pos'])
    rocket_scatter.set_offsets(offsets)
    rocket_scatter.set_color(current_colors)
    
    frame_counter += 1

    # move onto next generation if all rockets are dead or have won
    dead_count = sum(1 for r in rocket_pop if r['dead'])
    winners_count = sum(1 for r in rocket_pop if r['won'])

    # evolve population if all rockets are dead or have won, or if max frames for generation is reached
    if (dead_count + winners_count == num_rockets) or (frame_counter >= num_frames):
        
        # calculate fitness for all rockets and print generation stats
        fits = [fitness(r) for r in rocket_pop]
        best_idx = np.argmax(fits)

        # collect all winners (best individuals)
        winners = [r for r in rocket_pop if r['won']]

        print("-" * 40)
        print(f"GEN {generation + 1}")

        # number of winners
        print(f"Winners: {len(winners)} / {num_rockets}")

        # best fitness (will be 100 if any winner exists)
        print(f"Best Fitness: {fits[best_idx]:.2f}")

        # prints the mass and thrust of all winners
        if winners:
            print("Winner Mass/Thrust:")
            for r in winners:
                m = r['genome']['mass']
                t = r['genome']['thrust']
                print(f"  mass={m:.3f}, thrust={t:.3f}")
        else:
            print("No winners this generation")

        print("-" * 40)

        frame_counter = 0; # reset frame counter
        generation += 1 # next generation

        # stop simulation if max generations reached
        if generation >= num_generations:
            print("Simulation complete")
            ani.event_source.stop()
            return
        
        rocket_pop = evolve(rocket_pop)

    return [rocket_scatter] + traj_lines


# Run the animation
ani = FuncAnimation(fig, update, frames=None, interval=15, blit=False, cache_frame_data=False)
plt.show()