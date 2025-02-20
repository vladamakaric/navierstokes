#version 410
uniform ivec2 resolution;
uniform int columns;               
uniform int rows;                 
uniform usampler2D obstacleTexture;
uniform sampler2D fluidTexture;

out vec4 color;

// Not a full modulo, only covers single wraparound.
int modulo(int a, int b) {
    if (a < 0) {
        return b + a;
    }
    if (a >= b) {
        return a - b;
    }
    return a;
}

bool inObstacle(vec2 p) {
    ivec2 index = ivec2(int(floor(p.x)), int(floor(p.y)));
    if (texelFetch(obstacleTexture, index, 0).x == 0) {
        return false;
    }
    vec2 center = index + vec2(0.5);
    ivec2 dirs[4] = ivec2[](
        ivec2(0, 1), ivec2(1, 0), ivec2(0, -1), ivec2(-1, 0)
    );
    ivec2 normal = ivec2(0,0);
    for (int i = 0; i < 4; i++) {
        ivec2 neighbor = ivec2(
            modulo(index.x + dirs[i].x, columns),
            modulo(index.y + dirs[i].y, rows)
        );
        if (texelFetch(obstacleTexture, neighbor, 0).x == 0) {
            normal += dirs[i];
        }
    }
    // Horizontal or vertical edge.
    if (normal.x == 0 || normal.y == 0) {
        return true;
    }
    // 45 degree Corner.
    if (dot(p - center, normal) < 0) {
        return true;
    }
    return false;
} 

void main() {
    vec2 p = (gl_FragCoord.xy / vec2(resolution)) * vec2(columns,rows);
    if (inObstacle(p) == true) {
        discard;
    }
    color = vec4(texture(fluidTexture, gl_FragCoord.xy/vec2(resolution)).xyz,1.0);
}