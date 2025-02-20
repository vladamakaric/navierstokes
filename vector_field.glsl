// Based on https://www.shadertoy.com/view/4tc3DX
// TODO: Peep this shader for a time-varying vector field: https://www.shadertoy.com/view/4s23DG
#version 410
uniform vec2 resolution;           // viewport resolution (in pixels)
uniform vec2 mouse_px;           // viewport resolution (in pixels)
uniform vec2 mouse_box_size_px;           // viewport resolution (in pixels)
uniform int columns;               
uniform int rows;                 
out vec4 fragColor;
uniform sampler2D vectorFieldTexture;
uniform usampler2D obstacleTexture;
uniform sampler2D noiseTexture;
// Clamp [0..1] range
#define saturate(a) clamp(a, 0.0, 1.0)


const float PI = 3.1415926535897932384626433832795;
const float cell_size = 1.0;

float triangleWave(float x) { return abs(fract(x)-0.5)*2.0; }

// This is it... what you have been waiting for... _The_ Glorious Line Algorithm.
// This function will make a signed distance field that says how far you are from the edge
// of the line at any point U,V.
// Pass it UVs, line end points, line thickness (x is along the line and y is perpendicular),
// How rounded the end points should be (0.0 is rectangular, setting rounded to thick.y will be circular),
float LineDistField(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded) {
    // Don't let it get more round than circular.
    rounded = min(thick.y, rounded);
    // midpoint
    vec2 mid = (pB + pA) * 0.5;
    // vector from point A to B
    vec2 delta = pB - pA;
    // Distance between endpoints
    float lenD = length(delta);
    // unit vector pointing in the line's direction
    vec2 unit = delta / lenD;
    // Check for when line endpoints are the same
    if (lenD < 0.0001) unit = vec2(1.0, 0.0);	// if pA and pB are same
    // Perpendicular vector to unit - also length 1.0
    vec2 perp = unit.yx * vec2(-1.0, 1.0);
    // position along line from midpoint
    float dpx = dot(unit, uv - mid);
    // distance away from line at a right angle
    float dpy = dot(perp, uv - mid);
    // Make a distance function that is 0 at the transition from black to white
    float disty = abs(dpy) - thick.y + rounded;
    float distx = abs(dpx) - lenD * 0.5 - thick.x + rounded;

    // Too tired to remember what this does. Something like rounded endpoints for distance function.
    float dist = length(vec2(max(0.0, distx), max(0.0,disty))) - rounded;
    dist = min(dist, max(distx, disty));
    return dist;
}

// This makes a line in UV units. A 1.0 thick line will span a whole 0..1 in UV space.
float FillLine(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded) {
    float df = LineDistField(uv, pA, pB, vec2(thick), rounded);
    return saturate(df / abs(dFdy(uv).y));
}

float normalizationFactor() {
    return rows / resolution.y;
}

ivec2 cellIndex(vec2 p) {
    return ivec2(int(floor(p.x/cell_size)), int(floor(p.y/cell_size)));
}

float GridLines(vec2 p) {
    // Both of these are maximally 0.4.
    float horizontal_grid_line = saturate(triangleWave(p.x/cell_size) - 0.95)*8.0;
    float vertical_grid_line = saturate(triangleWave(p.y/cell_size) - 0.95)*8.0;
    return 1.0-max(horizontal_grid_line, vertical_grid_line);
}

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

float inObstacle(vec2 p) {
    // TODO: Implement real distance within obstacle (not just 0,1), for antialiasing.
    ivec2 index = cellIndex(p);
    if (texelFetch(obstacleTexture, index, 0).x == 0) {
        return 0;
    }
    vec2 center = vec2(
        index.x*cell_size + cell_size/2,
        index.y*cell_size + cell_size/2
    );
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
        return 1;
    }
    // 45 degree Corner.
    if (dot(p - center, normal) < 0) {
        return 1;
    }
    return 0;
} 


float MouseBox(vec2 p) {
    vec2 m = mouse_px * normalizationFactor();
    vec2 bs = mouse_box_size_px * normalizationFactor();
    return min(
        min(
            // Vertical sides
            FillLine(p, m+bs/2,m + vec2(bs.x, -bs.y)/2, vec2(0.0,0.01), 0.0),
            FillLine(p, m-bs/2,m - vec2(bs.x, -bs.y)/2, vec2(0.0,0.01), 0.0)
        ),
        min(
            // Horizontal sides
            FillLine(p, m+bs/2,m + vec2(-bs.x, bs.y)/2, vec2(0.0,0.01), 0.0),
            FillLine(p, m-bs/2,m - vec2(-bs.x, bs.y)/2, vec2(0.0,0.01), 0.0)
        )
    );
}

float Arrow(vec2 p, vec2 a, vec2 b, float thickness, float arrowHeadLengthAlongArrow, float angle) {
    // TODO: This is not an ideal arrow, the three lines converging at the top do
    // not give a nice sharp point. Need a special distance field, probably best
    // to make it for a triangle, and stick that on top of the arrow.

    // An arrow is an isoceles triangle, it has two equal length legs and a base.
    float legLength = arrowHeadLengthAlongArrow/cos(angle);
    float baseLength = 2*legLength*sin(angle);

    vec2 arrowDir = (b-a)/length(b-a);
    vec2 perpDir = vec2(-arrowDir.y, arrowDir.x);

    vec2 basePointOnShaft = b - arrowDir*arrowHeadLengthAlongArrow;
    vec2 baseEnd1 = basePointOnShaft + perpDir*baseLength/2;
    vec2 baseEnd2 = basePointOnShaft - perpDir*baseLength/2;

    vec3 color = vec3(1.0);
    float leg1 = FillLine(p, baseEnd1, b, vec2(0.0,thickness), 0.0);
    float leg2 = FillLine(p, baseEnd2, b, vec2(0.0,thickness), 0.0);
    float shaft = FillLine(p, a, b, vec2(0.0,thickness), 0.0);
    return min(min(leg1,leg2), shaft);
}

float VectorFieldArrows(vec2 p) {
    ivec2 cell_index = cellIndex(p);
    vec2 vector = texelFetch(vectorFieldTexture, cell_index, 0).xy;
    vec2 center = vec2(
        cell_index.x*cell_size + cell_size/2,
        cell_index.y*cell_size + cell_size/2
    );
    vec2 end = center + vector*cell_size/2;
    if (length(center-end) > 0.001) {
        return Arrow(p, center, end, 0.009, (cell_size/2)*0.3, PI/6);
    }
    return 1;
}

struct ColorSum {
    vec3 sum;
    int num_samples;
};

ColorSum sumColorAlongStreamline(vec2 start, int dir, float step_size, int max_num_steps) {
    vec2 p_size = resolution * normalizationFactor();
    vec3 sum = vec3(0.0);
    vec2 curr_pos = start;
    int num_steps = 0;
    float dt = 4;
    float totalDt = 0;
    float one_pixel_size = columns/800.0;
    for (float i = 0.; i < max_num_steps; i++) {
        // Runge Kutta 2
        vec2 v = dir * texture(vectorFieldTexture, curr_pos / p_size, 0).xy;

        float vl = length(v);

        // vec2 v = dir * texture(vectorFieldTexture, curr_pos / p_size, 0).xy;

        // Velocity gradient
        float uleft = texture(vectorFieldTexture, (curr_pos - vec2(one_pixel_size, 0)) / p_size, 0).x;
        float uright = texture(vectorFieldTexture, (curr_pos + vec2(one_pixel_size, 0)) / p_size, 0).x;
        float vdown = texture(vectorFieldTexture, (curr_pos - vec2(0, one_pixel_size)) / p_size, 0).y;
        float vup = texture(vectorFieldTexture, (curr_pos + vec2(0, one_pixel_size)) / p_size, 0).y;
        vec2 grad = vec2((uright - uleft)/(2*one_pixel_size), (vup - vdown)/(2*one_pixel_size));
        ///////
// length(grad));

        float dtt = one_pixel_size/(vl+4*length(grad));
        // float dtt = one_pixel_size/vl;
        
        vec2 pos_mid = curr_pos + v * dtt / 2;
        vec2 v_mid = dir * texture(vectorFieldTexture, pos_mid / p_size, 0).xy;
        // TODO: Depending on this v_mid, take dt small enough that I don't overshoot.
        // But actually I need to subsample according to how quickly the velocity
        // field is changing. I need to do the same thing in the advection step.
        curr_pos += v_mid * dtt;
        // curr_pos += v * dtt;;
        totalDt += dtt;
        // TODO: Stop at boundaries and obstacles.
        if (curr_pos.x < 0 || curr_pos.x > p_size.x || curr_pos.y < 0 || curr_pos.y > p_size.y) {
            break;
        }
        // if (inObstacle(curr_pos) > 0){
        //     break;
        // }
        sum += texture(noiseTexture, curr_pos / p_size).xyz;
        num_steps+=1;
        if (totalDt > dt) {
            break;
        }
    }
    return ColorSum(sum, num_steps);
}

float adjustContrast(float x, float k) {
    return 1.0 / (1.0 + exp(-k * (x - 0.5)));
}

vec3 AvgTextureAlongVelocityField(vec2 p) {
    // Also known as 'Line Integral Convolution'.
    // TODO: Adaptive stepsize.
    if (inObstacle(p) > 0) {
        return vec3(0.0);
    } 
    vec2 p_size = resolution * normalizationFactor();
    vec3 p_color = texture(noiseTexture, p / p_size).xyz;
    int max_num_steps = 100;
    float step_size = 0.01;
    ColorSum forward = sumColorAlongStreamline(p, 1, step_size, max_num_steps);
    ColorSum backward = sumColorAlongStreamline(p, -1, step_size, max_num_steps);
    vec3 sum = p_color + forward.sum + backward.sum;
    // vec3 sum = p_color + forward.sum;
    // TODO: Try to increase the contrast on the result.
    vec3 color = sum / (1 + forward.num_samples + backward.num_samples);
    // vec3 color = sum / (1 + forward.num_samples);
    float steepness = 10;
    return vec3(
        adjustContrast(color.r, steepness),
        adjustContrast(color.g, steepness),
        adjustContrast(color.b, steepness));
}

void main() {
    float normalizationFactor = normalizationFactor();
    vec2 p = gl_FragCoord.xy * normalizationFactor;
    vec2 p_size = resolution * normalizationFactor;
    if (p.x > cell_size*columns || p.y > cell_size*rows){
        discard;
    }
    vec3 finalColor = AvgTextureAlongVelocityField(p);
    // vec3 finalColor = vec3(1.0,1.0,1.0);
    // finalColor *= VectorFieldArrows(p);
    float finalColor5 = VectorFieldArrows(p);
    // TODO: get rid of this. And just press f to apply force.
    finalColor *= (1-MouseBox(p)*0.001);
    finalColor *= GridLines(p);

    ivec2 index = cellIndex(p);

    if (inObstacle(p) > 0) {
        fragColor = vec4(0.0,0.0,0.0,1.0);
    } else {
        fragColor = vec4(1.0,1.0,1.0,0.0);
    }
    
    // fragColor = vec4(finalColor, 1.0);
    // fragColor = vec4(sqrt(finalColor), 1.0);
}