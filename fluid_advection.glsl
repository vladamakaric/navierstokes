#version 410
uniform ivec2 resolution;           // viewport resolution (in pixels)
uniform int columns;               
uniform int rows;                 
uniform float dt;                 
uniform sampler2D vectorFieldTexture;
// uniform usampler2D obstacleTexture;
uniform sampler2D fluidTexture;

out vec4 color;

void main() {
    vec2 p = (gl_FragCoord.xy / vec2(resolution)) * vec2(columns,rows);
    int steps = 10;
    for (int i=0; i< steps; i++) {
        vec2 v = -texture(vectorFieldTexture, p / vec2(columns,rows), 0).xy;
        p += v*dt/steps;
    }

    // TODO: Hand-roll bilinear interpolation of velocity and the texture.

    // vec2 v = vec2(10,0);
    // float rdt = sin(dt*100);
    // vec2 p_mid = p - v1*dt/2;
    // vec2 v_mid =texture(vectorFieldTexture, p_mid / vec2(columns,rows), 0).xy;
    // vec2 p_back = p - v_mid*dt;
    color = texture(fluidTexture, p / vec2(columns,rows));
}