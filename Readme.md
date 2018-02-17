# NNoptExpt

This project is used for checking different types of optimizations when using deep learning algorithms. Deep learning algorithms typically have some form of gradient-decent algorithm used for fitting available data. However, it turns out that there are a multitude of algorithms that can be used for optimization, some of which are gradient-based, and others are not encumbered by the vagaries of saddle points and other irregular geometry. Especially useful are evolutionary algorithms that have ben shown to be less suceptible to converging toward local minima. 

## Getting Started

Just clone the repository and follow instructions in the Installation section. This section will be expanded as the program is written. 

## Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

## Installing

The folloiwing installations are for \*nix-like systems. These have been tried on macOS Sierra (Version 10.12.6) before. 

1. Clone the program to your computer. 
2. type `make firstRun`. This should do the following
    2.1. generate a virtual environment in folder `env`
    2.2. install a number of packages
    2.3. generate a new `requirements.txt` file
    2.4. generate an initial git repository
3. change to the `src` folder
4. run the command `make run`. This should run the small test program
5. Generate your documentation folder by running `make doc`. 
6. Check whether all the tests pass with the command `make test`. This uses py.test for running tests. 

## Deployment

This program is not meant to be deployed ...

## Built With

 - Python 3.6

## Contributing

Please send in a pull request.

## Authors

Sankha Mukherjee - Initial work (2018)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

 - Hat tip to anyone who's code was used
 - Inspiration
 - etc.
 