.. _run_tests:

Run HTTomo tests
----------------

After installing HTTomo, you can quickly verify that the installation and all required dependencies are working correctly by running the test suite.

1. Git clone the HTTomo repository :code:`git clone https://github.com/DiamondLightSource/httomo.git`.

2. Install the testing dependencies: :code:`conda install -c conda-forge pytest pytest-cov pytest-xdist pytest-mock plumbum`.

3. **Run the CPU test suite.** Navigate to the root directory of the HTTomo repository and run: :code:`pytest tests/`. This executes the CPU-only tests for the HTTomo framework.

4. **Run the GPU test suite (CUDA-enabled systems only).** If you have a CUDA-compatible GPU, run: :code:`pytest tests/ --cupy`.

5. **Run the small dataset pipeline tests.** :code:`pytest tests/ --small_data`. These tests execute example pipelines using a small test dataset. On systems without a CUDA-compatible GPU, some tests are expected to fail. However, if TomoPy is installed, the TomoPy pipeline test should pass successfully.

