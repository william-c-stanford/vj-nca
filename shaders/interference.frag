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
