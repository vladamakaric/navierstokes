#version 410
uniform ivec2 resolution;
uniform int columns;               
uniform int rows;                 
uniform float dt;                 
uniform sampler2D vectorFieldTexture;
uniform sampler2D fluidTexture;

out vec4 color;

void main() {
    vec2 p = (gl_FragCoord.xy / vec2(resolution)) * vec2(columns,rows);
    // TODO: Ideally this backward particle trace should be adaptive,
    // finer grained in regions of large velocity change (|grad V|).
    int steps = 10;
    for (int i=0; i< steps; i++) {
        vec2 v = texture(vectorFieldTexture, p / vec2(columns,rows), 0).xy;
        vec2 p_mid = p - v*dt/(steps*2);
        vec2 v_mid = texture(vectorFieldTexture, p_mid / vec2(columns,rows), 0).xy;
        p += -v_mid*dt/steps;
    }
    color = texture(fluidTexture, p / vec2(columns,rows));
}