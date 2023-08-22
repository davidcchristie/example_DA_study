
cartesian = True
if cartesian:

    # Number of particles in the initial distribution
    
    d_config_particles["n_x"] = 200
    d_config_particles["delta_x"] = (d_config_particles["r_max"] - d_config_particles["r_min"])/d_config_particles["n_x"]


    # see how many particles you actually will be working with
    nOuterSquare = d_config_particles["n_x"]/d_config_particles["delta_x"]
    linXY = np.linspace(0, d_config_particles["r_max"], nOuterSquare)
    [X,Y] = np.meshgrid(linXY, linXY)
    R = np.sqrt(X**2 + Y**2)
    # how many particles have R <= r_max and R >= r_min?
    nParticles = np.sum((R <= d_config_particles["r_max"]) & (R >= d_config_particles["r_min"]))
    print("nParticles = ", nParticles)
    # ask user whether this is OK
    bool_inp = input("Is this OK? (y/n): ")
    if bool_inp == "n":
        






    # create linspace with 
    
    # Number of angles for the initial particle distribution
    # d_config_particles["n_angles"] = 24

else:
