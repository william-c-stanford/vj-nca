// APERTURE: rounded rectangular light contours travelling through a fixed opening.
// The corner geometry stays simple; phase changes contour position, not exposure.
void main() {
    vec2 p = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
    float t = TAU * u_phase;
    vec2 q = rotate2(u_tilt) * p;
    q.x += u_warp * 0.30 * sin(t) * q.y;
    q.y += u_warp * 0.16 * cos(t);
    float d = boxSDF(q, vec2(0.27, 0.32)) - 0.16;
    // One contour advances per loop; the small periodic speed change creates a phrase.
    float carrier = d - (u_phase + 0.06 * sin(t)) / u_density;
    float contours = lineTrain(carrier, u_density, u_width);
    float outer = inside(boxSDF(q, vec2(0.77, 0.78)));
    float opening = smoothstep(-0.17, -0.14, d);
    float light = contours * outer * opening;
    float accent = smoothstep(0.0, 0.55, q.x) * smoothstep(-0.4, 0.1, q.y);
    fragColor = vec4(finishLight(light, accent, p), 1.0);
}
