Implementation of real time fluid dynamics based on Jos Stam's "Stable Fluids" paper.

I've implemented flow around obstacles, but only simple ones with 45 and 90 degree edges, like bullets and valves.

Here's a youtube video of flow around a bullet:

[![video demo](https://img.youtube.com/vi/NhceruVWtdM/hqdefault.jpg)](https://www.youtube.com/watch?v=NhceruVWtdM)

All the simulation logic is in python (mostly numpy), and all the rendering is in OpenGL shaders (see the rendering directory).

For a more detailed description of the simulator, see the documentation in simulator.py.

