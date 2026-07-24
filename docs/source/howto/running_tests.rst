.. _run_tests:

Run HTTomo tests
----------------

Running tests after installation is a safe way to test quickly if installation of HTTomo and dependencies is successful. 

1. Git clone HTTomo repository :code:`git clone https://github.com/DiamondLightSource/httomo.git`
2. Install testing dependencies: :code:`conda install -c conda-forge pytest pytest-cov pytest-xdist pytest-mock plumbum`
3. Go to main HTTomo folder and run :code:`pytest tests/`. This will run the HTTomo **framework** CPU tests only. 
4. If you have a GPU (CUDA-enabled), run GPU tests :code:`pytest tests/ --cupy`.
5. Run simple pipeline tests with a small test data :code:`pytest tests/ --small_data`. Note that if you don't have a GPU, some of the tests fail for you, however the TomoPy test (if TomoPy installed) must pass. 

