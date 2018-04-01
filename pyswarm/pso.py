from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import logging

class Pso(object):
    """
    Particle Swarm Optimization class
    """
    def __init__(self, maxiter=100, minstep=1e-6, minfunc=1e-6,
                debug=False, verbose=False, particle_output=False):
        """
        Instantiation
        :param maxiter: int
            The maximum number of iterations for the swarm to search (Default: 100)
        :param minstep: scalar
            The minimum stepsize of swarm's best position before the search
            terminates (Default: 1e-8)
        :param minfunc: scalar
            The minimum change of swarm's best objective value before the search
            terminates (Default: 1e-8)
        :param debug: boolean
            If True, error messages will be displayed
            (Default: False)
        :param verbose:  boolean
            If True, progress statements will be displayed every iteration
            (Default: True)
        :param particle_output: boolean
            get the full convergence results
            (Default: False)
        """
        self.maxiter = maxiter
        self.minstep = minstep
        self.minfunc = minfunc
        self.debug = debug
        self.verbose = verbose
        self.particle_output = particle_output
        self.__ready = False
        
    def initialize(self, func, lb, ub, ieqcons=[], f_ieqcons=None, 
                    args=(), kwargs={}):
        """
        Initialize the information on the variables and function
        :param func: function name to optimize
        :param lb: list
            lower bounds of the design variables
        :param ub: list
            upper bounds of the design variables
        :param ieqcons: list
            A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in
            a successfully optimized problem (Default: [])
        :param f_ieqcons: function
            Returns a 1-D array in which each element must be greater or equal
            to 0.0 in a successfully optimized problem. If f_ieqcons is specified,
            ieqcons is ignored (Default: None)
        :param args: tuple
            Additional arguments passed to objective and constraint functions
            (Default: empty tuple)
        :param kwargs: dict
            Additional keyword arguments passed to objective and constraint
            functions (Default: empty dict)
        :return: 0 (success value) or None if error
        """
        if 'function' not in str(type(func)):
            if self.debug:
                logging.debug( '--> PSO Error!' )
                logging.debug( '\tUnable to evaluate the function' )
            return None
        self.func = func
        self.lb = lb
        self.ub = ub
        self.ieqcons = ieqcons
        self.f_ieqcons = f_ieqcons
        self.args = args
        self.kwargs = kwargs
        self.__ready = True
        return 0
    
    def _obj_wrapper(self, func, args, kwargs, x):
        return func(x, *args, **kwargs)

    def _is_feasible_wrapper(self, func, x):
        return np.all(func(x)>=0)

    def _cons_none_wrapper(self, x):
        return np.array([0])

    def _cons_ieqcons_wrapper(self, ieqcons, args, kwargs, x):
        return np.array([y(x, *args, **kwargs) for y in ieqcons])

    def _cons_f_ieqcons_wrapper(self, f_ieqcons, args, kwargs, x):
        return np.array(f_ieqcons(x, *args, **kwargs))
        
    def optimize(self, swarmsize=100, omega=0.5, phip=0.5, phig=0.5, processes=1):
        """
        Perform a particle swarm optimization (PSO)
        :param swarmsize: int
            The number of particles in the swarm (Default: 100)
        :param omega: scalar
            Particle velocity scaling factor (Default: 0.5)
        :param phip: scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        :param phig: scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        :param processes: int
            Choice for parallelism:            
            = 1: a sequential computation is done
            > 1: pathos module is used for multiprocessing
            = 0: the set of particles is given to the cost function self.func for a user implemented parallelism
        :return: results of the optimal solution. None if error encountered.
            g: array
                The swarm's best known position (optimal design)
            f: scalar
                The objective value at ``g``
            p: array
                The best known position per particle
            fp: array
                The objective values at each position in p
            p_save: array
                The convergence of all the particles (positions)
            fp_save: array
                The convergence of all the particles (cost function)
        """
        if self.verbose:
            level = logging.INFO
        elif self.debug:
            level = logging.DEBUG
        logging.basicConfig(filename=__name__+'_log.txt',level=level,\
            format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')
        if self.__ready is False:
            if self.debug:
                logging.debug( '--> PSO Error!' )
                logging.debug( '\tThe problem is not correctly initialized.' )
                logging.debug( '\tRun the "initialize" method before optimizing.' )
            return None
        if len(self.lb) != len(self.ub):
            logging.debug('Lower- and upper-bounds must be the same length')
        if hasattr(self.func, '__call__') == False:
            logging.debug('Invalid function handle')
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)
        if np.all(self.ub>self.lb) == True:
        	logging.debug('All upper-bound values must be greater than lower-bound values')
       
        vhigh = np.abs(self.ub - self.lb)
        vlow = -vhigh

        # Initialize objective function
        self.obj = partial(self._obj_wrapper, self.func, self.args, self.kwargs)
        
        # Check for constraint function(s)
        if self.f_ieqcons is None:
            if not len(self.ieqcons):
                if self.debug:
                    logging.debug('No constraints given.')
                self.cons = self._cons_none_wrapper
            else:
                if self.debug:
                    logging.debug('Converting ieqcons to a single constraint function')
                self.cons = partial(self._cons_ieqcons_wrapper, self.ieqcons, self.args, self.kwargs)
        else:
            if self.debug:
                logging.debug('Single constraint function given in f_ieqcons')
            self.cons = partial(self._cons_f_ieqcons_wrapper, self.f_ieqcons, self.args, self.kwargs)
        self.is_feasible = partial(self._is_feasible_wrapper, self.cons)

        # Initialize the particle swarm
        S = swarmsize
        D = len(self.lb)  # the number of dimensions each particle has
        x = np.random.rand(S, D)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(S)  # current particle function values
        fs = np.ones(S, dtype=bool)  # feasibility of each particle
        fp = np.ones(S)*np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value
        
        # Initialize the particle's position
        x = self.lb  + x*(self.ub - self.lb)

        # Create a pool of processors
        if processes > 1:
            pool = Pool(nodes=processes)
            # Compute objective and constraints for each particle
            fx = np.array(pool.map(self.func, x))
            fs = np.array(pool.map(self.is_feasible, x))
        elif processes == 1:
            for i in range(S):
                fx[i] = self.obj(x[i, :])
                fs[i] = self.is_feasible(x[i, :])
        else:
            fx = self.func(x)
            fs = self.is_feasible(x)

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Store all the convergence phase
        p_save = x # particles
        fp_save = fx # function values

        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()
           
        # Initialize the particle's velocity
        v = vlow + np.random.rand(S, D)*(vhigh - vlow)

        # Iterate until termination criterion met
        it = 1
        while it <= self.maxiter:
            rp = np.random.uniform(size=(S, D))
            rg = np.random.uniform(size=(S, D))

            # Update the particles velocities
            v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < self.lb
            masku = x > self.ub
            x = x*(~np.logical_or(maskl, masku)) + self.lb*maskl + self.ub*masku

            # Update objectives and constraints
            if processes > 1:
                fx = np.array(pool.map(self.func, x))
                # bug can't pickle: fs = np.array(pool.map(self.is_feasible, x))
            elif processes == 1:
                for i in range(S):
                    fx[i] = self.func(x[i, :])
                    fs[i] = self.is_feasible(x[i, :])
            else:
                fx = self.func(x)
                fs = self.is_feasible(x)

            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]

            # Store the current solution
            p_save = np.vstack((p_save, p))
            fp_save = np.hstack((fp_save, fp))

            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                if self.verbose:
                    logging.info('New best for swarm at iteration {:}: {:} {:}'\
                        .format(it, p[i_min, :], fp[i_min]))

                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))

                if np.abs(fg - fp[i_min]) <= self.minfunc:
                    if self.verbose:
                        logging.info('Stopping search: Swarm best objective change less than {:}'\
                        .format(self.minfunc))
                    if self.particle_output:
                        return{'optimal particle':p_min, 'optimal function':fp[i_min], \
                        'convergence particles':p_save, 'convergence functions':fp_save}
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= self.minstep:
                    if self.verbose:
                        logging.info('Stopping search: Swarm best position change less than {:}'\
                        .format(self.minstep))
                    if self.particle_output:
                        return{'optimal particle':p_min, 'optimal function':fp[i_min], \
                        'convergence particles':p_save, 'convergence functions':fp_save}
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]

            if self.verbose:
                logging.info('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1

        # Close the pool
        if processes > 1:
            pool.terminate()
            pool.join()

        if self.verbose:
            logging.info('Stopping search: maximum iterations reached --> {:}'.format(self.maxiter))
        
        if not self.is_feasible(g):
            if self.verbose:
                logging.info("However, the optimization couldn't find a feasible design. Sorry")
        if self.particle_output:
            if self.verbose:
                logging.info('The number of iterations is: ' + str(it))
            return{'optimal particle':g, 'optimal function':fg, \
                'convergence particles':p_save, 'convergence functions':fp_save}
        else:
            if self.verbose:
                logging.info('The number of iterations is: ' + str(it))
            return g, fg
