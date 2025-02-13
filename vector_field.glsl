// Based on https://www.shadertoy.com/view/4tc3DX
// TODO: Peep this shader for a time-varying vector field: https://www.shadertoy.com/view/4s23DG
#version 410
uniform vec2 resolution;           // viewport resolution (in pixels)
uniform vec2 mouse;           // viewport resolution (in pixels)
uniform vec2 mouseBoxSize;           // viewport resolution (in pixels)
uniform int columns;               
uniform int rows;                 
out vec4 fragColor;
uniform sampler2D vectorFieldTexture;
uniform usampler2D obstacleTexture;
uniform sampler2D noiseTexture;
// Clamp [0..1] range
#define saturate(a) clamp(a, 0.0, 1.0)


const float PI = 3.1415926535897932384626433832795;

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


vec3 DrawArrow(vec2 uv, vec2 a, vec2 b, float thickness, float arrowHeadLengthAlongArrow, float angle) {
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
    color *= FillLine(uv, a, b, vec2(0.0,thickness), 0.0);
    color *= FillLine(uv, baseEnd1, b, vec2(0.0,thickness), 0.0);
    color *= FillLine(uv, baseEnd2, b, vec2(0.0,thickness), 0.0);
    return color;
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    float normalizationFactor = 16.0 / resolution.y;
    vec2 uvSize = resolution * normalizationFactor;
    vec2 uv = fragCoord * normalizationFactor;
    float cellSize = min(uvSize.x / columns, uvSize.y / rows);
    if (uv.x > cellSize*columns || uv.y > cellSize*rows){
        discard;
    }

    ///// 
    int max_num_steps = 20;
    float step_size = 0.1;
    vec2 currPos = uv;
    vec3 colorAccum = texture(noiseTexture, currPos / uvSize).xyz;
    int num_steps = 0;
    // Forward steps
    // TODO: Adaptive stepsize.
    for (float i = 0.; i < max_num_steps; i++) {
        vec2 velocity = texture(vectorFieldTexture, currPos / uvSize, 0).xy;
        currPos += velocity*step_size;
        if (currPos.x < 0 || currPos.x > uvSize.x || currPos.y < 0 || currPos.y > uvSize.y) {
            break;
        }
        colorAccum += texture(noiseTexture, currPos / uvSize).xyz;
        num_steps+=1;
    }
    // Backward steps
    for (float i = 0.; i < max_num_steps; i++) {
        vec2 velocity = texture(vectorFieldTexture, currPos / uvSize, 0).xy;
        currPos -= velocity*step_size;
        if (currPos.x < 0 || currPos.x > uvSize.x || currPos.y < 0 || currPos.y > uvSize.y) {
            break;
        }
        colorAccum += texture(noiseTexture, currPos / uvSize).xyz;
        num_steps+=1;
    }
    colorAccum = colorAccum / (num_steps + 1);

    ////
    vec3 finalColor = colorAccum;

    int j = int(floor(uv.y/cellSize));
    int i = int(floor(uv.x/cellSize));
    vec2 vector = texelFetch(vectorFieldTexture, ivec2(i,j), 0).xy;
    vec2 center = vec2(i*cellSize + cellSize/2, j*cellSize + cellSize/2);
    vec2 end = center + vector*cellSize/2;

    // Drawing 

    if (texelFetch(obstacleTexture, ivec2(i,j), 0).x == 1) {
        finalColor.b = 0.9;
    }
    // finalColor *= FillLine(uv, center, end, vec2(0.0,0.02), 0.0);
    if (length(center-end) > 0.001) {
        finalColor *= DrawArrow(uv, center, end, 0.009, (cellSize/2)*0.3, PI/6);
    }


    // fragColor = texture(noiseTexture, gl_FragCoord.xy / resolution.xy);
    // fragColor = 

    finalColor *= colorAccum;
    // finalColor = colorAccum;

    vec2 mouseUv = mouse * normalizationFactor;
    vec2 bs = mouseBoxSize * normalizationFactor;
    // Vertical sides
    finalColor *= FillLine(uv, mouseUv+bs/2,mouseUv + vec2(bs.x, -bs.y)/2, vec2(0.0,0.01), 0.0);
    finalColor *= FillLine(uv, mouseUv-bs/2,mouseUv - vec2(bs.x, -bs.y)/2, vec2(0.0,0.01), 0.0);
    // Horizontal sides
    finalColor *= FillLine(uv, mouseUv+bs/2,mouseUv + vec2(-bs.x, bs.y)/2, vec2(0.0,0.01), 0.0);
    finalColor *= FillLine(uv, mouseUv-bs/2,mouseUv - vec2(-bs.x, bs.y)/2, vec2(0.0,0.01), 0.0);

    // colorAccum.xyz 
    // At most 0.2, but 0.0 if not on line.
    finalColor -= vec3(1.0, 1.0, 0.2) * saturate(triangleWave(uv.x/cellSize) - 0.95)*4.0;
    finalColor -= vec3(1.0, 1.0, 0.2) * saturate(triangleWave(uv.y/cellSize) - 0.95)*4.0;
    fragColor = vec4(sqrt(saturate(finalColor)), 1.0);

    // fragColor = colorAccum;

}