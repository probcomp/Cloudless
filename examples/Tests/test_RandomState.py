import numpy as np
import numpy.random as nr

n_rand = 1000
state_list = []
rand_list = []
for idx in range(n_rand):
    state_list.append(nr.get_state())
    rand_list.append(nr.rand())

nr.set_state(state_list[0])
rand_list_2 = nr.rand(n_rand)
assert all(np.array(rand_list)==np.array(rand_list_2)),"setting RandomState didn't work!"
print "Setting RandomState worked!"

# >>> help(np.random.RandomState)
# Help on class RandomState:

# class RandomState(__builtin__.object)
#  |  get_state(...)
#  |      get_state()
#  |      
#  |      Return a tuple representing the internal state of the generator.
#  |      
#  |      For more details, see `set_state`.
#  |      Returns
#  |      -------
#  |      out : tuple(str, ndarray of 624 uints, int, int, float)
#  |          The returned tuple has the following items:
#  |      
#  |          1. the string 'MT19937'.
#  |          2. a 1-D array of 624 unsigned integer keys.
#  |          3. an integer ``pos``.
#  |          4. an integer ``has_gauss``.
#  |          5. a float ``cached_gaussian``.
#  |      Notes
#  |      -----
#  |      `set_state` and `get_state` are not needed to work with any of the
#  |      random distributions in NumPy. If the internal state is manually altered,
#  |      the user should know exactly what he/she is doing.



#  |  set_state(...)
#  |      set_state(state)
#  |      
#  |      Set the internal state of the generator from a tuple.
#  |      
#  |      For use if one has reason to manually (re-)set the internal state of the
#  |      "Mersenne Twister"[1]_ pseudo-random number generating algorithm.
#  |      
#  |      Parameters
#  |      ----------
#  |      state : tuple(str, ndarray of 624 uints, int, int, float)
#  |          The `state` tuple has the following items:
#  |      
#  |          1. the string 'MT19937', specifying the Mersenne Twister algorithm.
#  |          2. a 1-D array of 624 unsigned integers ``keys``.
#  |          3. an integer ``pos``.
#  |          4. an integer ``has_gauss``.
#  |          5. a float ``cached_gaussian``.
#  |      
#  |      Returns
#  |      -------
#  |      out : None
#  |          Returns 'None' on success.
#  |      
#  |      See Also
#  |      --------
#  |      get_state
#  |      
#  |      Notes
#  |      -----
#  |      `set_state` and `get_state` are not needed to work with any of the
#  |      random distributions in NumPy. If the internal state is manually altered,
#  |      the user should know exactly what he/she is doing.
#  |      
#  |      For backwards compatibility, the form (str, array of 624 uints, int) is
#  |      also accepted although it is missing some information about the cached
#  |      Gaussian value: ``state = ('MT19937', keys, pos)``.
#  |      
#  |      References
#  |      ----------
#  |      .. [1] M. Matsumoto and T. Nishimura, "Mersenne Twister: A
#  |         623-dimensionally equidistributed uniform pseudorandom number
#  |         generator," *ACM Trans. on Modeling and Computer Simulation*,
#  |         Vol. 8, No. 1, pp. 3-30, Jan. 1998.
