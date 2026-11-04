// FOLD: a bounded ribbed light sheet, folded by one travelling periodic wave.
// density: rib count; warp: fold depth; tilt: composition; width: light thickness.
void main() {
    vec2 p = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
    vec2 q = rotate2(u_tilt) * (p - vec2(-0.04, 0.02));
    float t = TAU * u_phase;
    float bow = u_warp * sin(3.0 * q.y + t);
    float squeeze = 0.78 + 0.18 * cos(2.4 * q.y - t);
    float x = (q.x - bow) / squeeze;
    float ribs = lineTrain(x, u_density, u_width);
    float silhouette = inside(boxSDF(vec2(x, q.y), vec2(0.73, 0.72)));
    float edge = 1.0 - smoothstep(0.0, 0.025, abs(abs(x) - 0.73));
    float light = max(ribs, edge * 0.45) * silhouette;
    // A narrow colored rib carries direction; the rest of the sheet stays ivory.
    float accent = exp(-pow((x + 0.22) / 0.10, 2.0));
    fragColor = vec4(finishLight(light, accent, p), 1.0);
}
