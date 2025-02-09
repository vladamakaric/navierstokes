// TODO: Shader for a time-varying vector field: https://www.shadertoy.com/view/4s23DG
// From https://www.shadertoy.com/view/4tc3DX
#version 410
uniform vec2      iResolution;           // viewport resolution (in pixels)
uniform float     iTime;                 // shader playback time (in seconds)
out vec4 fragColor;
// Clamp [0..1] range
#define saturate(a) clamp(a, 0.0, 1.0)

// Basically a triangle wave
float repeat(float x) { return abs(fract(x*0.5+0.5)-0.5)*2.0; }

float triangleWave(float x) { return abs(fract(x)-0.5)*2.0; }

// This is it... what you have been waiting for... _The_ Glorious Line Algorithm.
// This function will make a signed distance field that says how far you are from the edge
// of the line at any point U,V.
// Pass it UVs, line end points, line thickness (x is along the line and y is perpendicular),
// How rounded the end points should be (0.0 is rectangular, setting rounded to thick.y will be circular),
// dashOn is just 1.0 or 0.0 to turn on the dashed lines.
float LineDistField(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded, float dashOn) {
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

    // This is for animated dashed lines. Delete if you don't like dashes.
    float dashScale = 2.0*thick.y;
    // Make a distance function for the dashes
    float dash = (repeat(dpx/dashScale + iTime)-0.5)*dashScale;
    // Combine this distance function with the line's.
    dist = max(dist, dash-(1.0-dashOn*1.0)*10000.0);

    return dist;
}

// This makes a filled line in pixel units. A 1.0 thick line will be 1 pixel thick.
float FillLinePix(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded) {
    float scale = abs(dFdy(uv).y);
    thick = (thick * 0.5 - 0.5) * scale;
    float df = LineDistField(uv, pA, pB, vec2(thick), rounded, 0.0);
    return saturate(df / scale);
}

// This makes an outlined line in pixel units. A 1.0 thick outline will be 1 pixel thick.
float DrawOutlinePix(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded, float outlineThick) {
    float scale = abs(dFdy(uv).y);
    thick = (thick * 0.5 - 0.5) * scale;
    rounded = (rounded * 0.5 - 0.5) * scale;
    outlineThick = (outlineThick * 0.5 - 0.5) * scale;
    float df = LineDistField(uv, pA, pB, vec2(thick), rounded, 0.0);
    return saturate((abs(df + outlineThick) - outlineThick) / scale);
}

// This makes a line in UV units. A 1.0 thick line will span a whole 0..1 in UV space.
float FillLine(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded) {
    float df = LineDistField(uv, pA, pB, vec2(thick), rounded, 0.0);
    return saturate(df / abs(dFdy(uv).y));
}

// This makes a dashed line in UV units. A 1.0 thick line will span a whole 0..1 in UV space.
float FillLineDash(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded) {
    float df = LineDistField(uv, pA, pB, vec2(thick), rounded, 1.0);
    return saturate(df / abs(dFdy(uv).y));
}

// This makes an outlined line in UV units. A 1.0 thick outline will span 0..1 in UV space.
float DrawOutline(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded, float outlineThick) {
    float df = LineDistField(uv, pA, pB, vec2(thick), rounded, 0.0);
    return saturate((abs(df + outlineThick) - outlineThick) / abs(dFdy(uv).y));
}

// This just draws a point for debugging using a different technique that is less glorious.
void DrawPoint(vec2 uv, vec2 p, inout vec3 col) {
    col = mix(col, vec3(1.0, 0.25, 0.25), saturate(abs(dFdy(uv).y)*8.0/distance(uv, p)-4.0));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Adjust UV space so it's a nice size and square.
	vec2 uv = fragCoord.xy / iResolution.xy;
	uv -= 0.5;
    uv.x *= iResolution.x / iResolution.y;
    // (-8,8)x(-8,8) coordinate space
    uv *= 16.0;

    // Make things that rotate with time.
    vec2 rotA = vec2(cos(iTime*0.82), sin(iTime*0.82));
    vec2 rotB = vec2(sin(iTime*0.82), -cos(iTime*0.82));
    // Make a bunch of line endpoints to use.
    vec2 pA = vec2(-4.0, 0.0) - 0.2*rotA;
    vec2 pB = vec2(4.0, 0.0) + rotA;
    vec2 pC = pA + vec2(0.0, 4.0);
    vec2 pD = pB + vec2(0.0, 4.0);
    // Debugging code
    //float df = LineDistField(uv, pA, pB, vec2(28.0 * dFdy(uv).y), 0.1, 0.0);
    //float df = DistField(uv, pA, pB, 25.000625 * dFdx(uv).x, 0.5);
    //vec3 finalColor = vec3(df*1.0, -df*1.0, 0.0);
    //finalColor = vec3(1.0) * saturate(df / dFdy(uv).y);
    //finalColor = vec3(1.0) * saturate((abs(df+0.009)-0.009) / dFdy(uv).y);

    // Clear to white.
    vec3 finalColor = vec3(1.0);

    // Lots of sample lines
    // 1 pixel thick regardless of screen scale.
    finalColor *= FillLinePix(uv, pA, pB, vec2(1.0, 1.0), 0.0);
    // Rounded rectangle outline, 1 pixel thick
    finalColor *= DrawOutlinePix(uv, pA, pB, vec2(32.0), 16.0, 1.0);
    // square-cornered rectangle outline, 1 pixel thick
    finalColor *= DrawOutlinePix(uv, pA, pB, vec2(64.0), 0.0, 1.0);
    // Fully rounded endpoint with outline 8 pixels thick
    finalColor *= DrawOutlinePix(uv, pA, pB, vec2(128.0), 128.0, 8.0);
    // Dashed line with rectangular endpoints that touch pC and pD, 0.5 radius thickness in UV units
    finalColor *= FillLineDash(uv, pC, pD, vec2(0.0, 0.5), 0.0);
    // Rounded endpoint dashed line with radius 0.125 in UV units
    finalColor *= FillLineDash(uv, pC + vec2(0.0, 2.0), pD + vec2(0.0, 2.0), vec2(0.125), 1.0);
    
    finalColor *= DrawOutline(uv, (pA + pB) * 0.5 + vec2(0.0, -4.5), (pA + pB) * 0.5 + vec2(0.0, -4.5), vec2(2.0, 2.0), 2.0, 0.8);
    finalColor *= FillLine(uv, pA - vec2(4.0, 0.0), pC - vec2(4.0, 0.0)+rotA, vec2(0.125), 0.0);
    finalColor *= FillLine(uv, pB + vec2(4.0, 0.0), pD + vec2(4.0, 0.0)-rotA, vec2(0.125), 1.0);

    DrawPoint(uv, pA, finalColor);
    DrawPoint(uv, pB, finalColor);
    DrawPoint(uv, pC, finalColor);
    DrawPoint(uv, pD, finalColor);

    // Blue grid lines
    finalColor -= vec3(1.0, 1.0, 0.2) * saturate(repeat(uv.x*2.0) - 0.92)*4.0;
    finalColor -= vec3(1.0, 1.0, 0.2) * saturate(repeat(uv.y*2.0) - 0.95)*4.0;
    // finalColor *= saturate(mod(fragCoord.y + 0.5, 2.0) + mod(fragCoord.x + 0.5, 2.0));
    fragColor = vec4(sqrt(saturate(finalColor)), 1.0);
}

void main() {
    // Use gl_FragCoord.xy as the fragment coordinate.
    mainImage(fragColor, gl_FragCoord.xy);
}