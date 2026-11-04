#version 330
// Original VJ studies, MIT code; rendered outputs CC0-1.0.
// All temporal arguments are periodic in u_phase; uniforms are the control data.
uniform vec2 u_resolution;
uniform float u_phase;
uniform float u_density;
uniform float u_warp;
uniform float u_tilt;
uniform float u_width;
uniform float u_accent;
uniform float u_softness;
out vec4 fragColor;
const float TAU = 6.283185307179586;

mat2 rotate2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

float boxSDF(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float lineTrain(float coordinate, float density, float width) {
    float d = abs(fract(coordinate * density + 0.5) - 0.5) / density;
    float aa = max(fwidth(coordinate), u_softness);
    return 1.0 - smoothstep(width, width + aa, d);
}

float inside(float distance) {
    float aa = max(fwidth(distance), 1.0 / u_resolution.y);
    return 1.0 - smoothstep(-aa, aa, distance);
}

vec3 finishLight(float light, float accent, vec2 p) {
    vec3 ivory = vec3(0.87, 0.90, 0.84);
    vec3 acid = vec3(0.69, 0.98, 0.28);
    vec3 c = ivory * clamp(light, 0.0, 1.0);
    c = mix(c, acid * clamp(light, 0.0, 1.0), clamp(accent * u_accent, 0.0, 1.0));
    // A fixed spatial falloff keeps the frame dark without global luminance pulsing.
    return c * (0.92 + 0.08 * exp(-dot(p, p)));
}
// INTERFERENCE: two finite line panels shear past each other without noisy feedback.
// Broad carrier lines and a bounded field make changing phase visibly trackable.
void main() {
    vec2 p = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
    float t = TAU * u_phase;
    vec2 q = rotate2(u_tilt) * p;
    float bend = u_warp * sin(2.5 * q.y + t);
    float carrier = q.x + bend;
    float a = lineTrain(carrier + 0.035 * sin(t), u_density, u_width);
    float b = lineTrain(carrier + 0.16 * q.y - 0.035 * sin(t), u_density, u_width);
    float panel = inside(boxSDF(q, vec2(0.70, 0.72)));
    // The second carrier becomes a moving cutout: broad dark interference bands.
    float light = a * (0.25 + 0.75 * b) * panel;
    float accent = smoothstep(0.28, 0.50, q.y) * 0.75;
    fragColor = vec4(finishLight(light, accent, p), 1.0);
}
