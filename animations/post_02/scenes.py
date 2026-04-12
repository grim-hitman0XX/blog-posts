"""Manim animations for Post 2: How One Data Point Steers the Entire Model.

Scenes:
  - PerSampleVotes: Data points each exert a force (arrow) on the ball.
    Forces combine into a net direction. Remove an outlier, watch the net swing.
  - CurvatureAmplifier: Two equal-length gradients on different curvature.
    H^{-1} amplifies the high-curvature-aligned one dramatically.
  - FingerprintDetector: Batch gradient decomposed via inner products with
    a library of per-sample gradients, recovering individual contributions.

NOTE: Requires a LaTeX installation for MathTex rendering.
"""

from manim import *
import numpy as np


# ──────────────────────────────────────────────
# Color palette (consistent with Post 1)
# ──────────────────────────────────────────────
COLORS = [BLUE, GREEN, ORANGE, PURPLE, RED, YELLOW, TEAL, PINK, MAROON, GOLD]
BG_COLOR = "#0f0f1a"
ACCENT = "#f0b429"
SOFT_WHITE = "#e0e0e8"
DIM_WHITE = "#888898"
PROPONENT = "#2ecc71"
OPPONENT = "#e74c3c"
BYSTANDER = "#95a5a6"


class PerSampleVotes(Scene):
    """Side-by-side comparison: two batches with the SAME average gradient
    but very different internal structure. Removing one data point from
    each reveals that disagreement matters — the outlier batch shifts
    dramatically, the consensus batch barely moves."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        # Title
        title = Text("Same Average, Different Story", font_size=36, color=SOFT_WHITE)
        subtitle = Text(
            "Two batches with identical gradients — but very different insides",
            font_size=20, color=DIM_WHITE,
        )
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.2)
        header.to_edge(UP, buff=0.3)
        self.play(FadeIn(header), run_time=0.6)

        # ── Helper: build a mini plane with ball and arrows ──
        def make_plane(center):
            plane = NumberPlane(
                x_range=[-2.5, 2.5, 1],
                y_range=[-2.5, 2.5, 1],
                x_length=5.5,
                y_length=4.5,
                background_line_style={
                    "stroke_color": GREY_D,
                    "stroke_width": 0.5,
                    "stroke_opacity": 0.15,
                },
                axis_config={"stroke_color": GREY_B, "stroke_width": 0.8},
            ).move_to(center)
            return plane

        # Two planes side by side
        left_center = LEFT * 3.3 + DOWN * 0.3
        right_center = RIGHT * 3.3 + DOWN * 0.3
        plane_L = make_plane(left_center)
        plane_R = make_plane(right_center)

        # Divider
        divider = DashedLine(
            UP * 2.5, DOWN * 3.0,
            color=GREY_D, stroke_width=1, dash_length=0.1,
        )

        label_L = Text("Batch A: disagreement", font_size=18, color=ORANGE).next_to(
            plane_L, UP, buff=0.15
        )
        label_R = Text("Batch B: consensus", font_size=18, color=TEAL).next_to(
            plane_R, UP, buff=0.15
        )

        self.play(
            Create(plane_L), Create(plane_R), Create(divider),
            FadeIn(label_L), FadeIn(label_R),
            run_time=0.8,
        )

        # ── Build the two batches ──
        # Target average gradient (same for both batches)
        target_avg = np.array([-0.45, -0.40])
        n = 5
        arrow_scale = 1.6

        # Batch A: 4 points pulling roughly the same way + 1 outlier
        # Outlier pulls very differently; the other 4 compensate so the
        # average still equals target_avg.
        outlier_a = np.array([0.35, 1.10])
        # remaining 4 must sum to n*target - outlier = 5*target - outlier
        remaining_sum = n * target_avg - outlier_a
        np.random.seed(7)
        grads_A = []
        for i in range(n - 1):
            base = remaining_sum / (n - 1)
            noise = np.random.normal(0, 0.08, 2)
            grads_A.append(base + noise)
        # Fix last one to ensure exact average
        grads_A[-1] = remaining_sum - sum(grads_A[:-1])
        grads_A.append(outlier_a)  # index 4 is the outlier

        # Batch B: all 5 points pulling roughly toward target_avg
        np.random.seed(11)
        grads_B = []
        for i in range(n):
            noise = np.random.normal(0, 0.06, 2)
            grads_B.append(target_avg + noise)
        # Fix last to ensure exact average
        current_sum = sum(grads_B[:-1])
        grads_B[-1] = n * target_avg - current_sum

        colors_A = [BLUE, GREEN, RED, PURPLE, ORANGE]
        colors_B = [BLUE, GREEN, RED, PURPLE, TEAL]

        # ── Draw arrows for both batches ──
        ball_L_pos = plane_L.c2p(0.8, 0.6)
        ball_R_pos = plane_R.c2p(0.8, 0.6)

        ball_L = Dot(ball_L_pos, color=ACCENT, radius=0.10)
        ball_L_glow = Dot(ball_L_pos, color=ACCENT, radius=0.16).set_opacity(0.25)
        ball_R = Dot(ball_R_pos, color=ACCENT, radius=0.10)
        ball_R_glow = Dot(ball_R_pos, color=ACCENT, radius=0.16).set_opacity(0.25)

        self.play(
            FadeIn(ball_L), FadeIn(ball_L_glow),
            FadeIn(ball_R), FadeIn(ball_R_glow),
            run_time=0.3,
        )

        arrows_L = VGroup()
        arrows_R = VGroup()

        for i in range(n):
            vec_a = grads_A[i] * arrow_scale
            end_a = ball_L_pos + np.array([vec_a[0], vec_a[1], 0])
            arr_a = Arrow(
                start=ball_L_pos, end=end_a,
                color=colors_A[i], stroke_width=2.5, buff=0,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows_L.add(arr_a)

            vec_b = grads_B[i] * arrow_scale
            end_b = ball_R_pos + np.array([vec_b[0], vec_b[1], 0])
            arr_b = Arrow(
                start=ball_R_pos, end=end_b,
                color=colors_B[i], stroke_width=2.5, buff=0,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows_R.add(arr_b)

        # Show all arrows
        self.play(
            *[GrowArrow(a) for a in arrows_L],
            *[GrowArrow(a) for a in arrows_R],
            run_time=1.0,
        )

        # Highlight the outlier in Batch A
        outlier_note = Text(
            "← outlier", font_size=16, color=ORANGE,
        ).next_to(arrows_L[4].get_end(), RIGHT, buff=0.1)
        self.play(FadeIn(outlier_note, shift=LEFT * 0.2), run_time=0.4)
        self.wait(0.6)
        self.play(FadeOut(outlier_note), run_time=0.3)

        # ── Phase 2: Show the resultants (identical!) ──
        mean_A = np.mean(grads_A, axis=0) * arrow_scale
        mean_B = np.mean(grads_B, axis=0) * arrow_scale

        res_L_end = ball_L_pos + np.array([mean_A[0], mean_A[1], 0])
        res_R_end = ball_R_pos + np.array([mean_B[0], mean_B[1], 0])

        resultant_L = Arrow(
            start=ball_L_pos, end=res_L_end,
            color=WHITE, stroke_width=5, buff=0,
            max_tip_length_to_length_ratio=0.2,
        )
        resultant_R = Arrow(
            start=ball_R_pos, end=res_R_end,
            color=WHITE, stroke_width=5, buff=0,
            max_tip_length_to_length_ratio=0.2,
        )

        same_text = Text(
            "Same average gradient",
            font_size=22, color=ACCENT,
        ).to_edge(DOWN, buff=0.4)

        self.play(
            *[a.animate.set_opacity(0.15) for a in arrows_L],
            *[a.animate.set_opacity(0.15) for a in arrows_R],
            GrowArrow(resultant_L), GrowArrow(resultant_R),
            FadeIn(same_text, shift=UP * 0.2),
            run_time=1.0,
        )
        self.wait(1.2)

        # ── Phase 3: Remove one point from each ──
        remove_text = Text(
            "Now remove one data point from each...",
            font_size=22, color=SOFT_WHITE,
        ).to_edge(DOWN, buff=0.4)

        self.play(
            ReplacementTransform(same_text, remove_text),
            # Fade out the outlier from A (index 4)
            arrows_L[4].animate.set_opacity(0.0),
            # Fade out the first point from B (index 0) — a typical one
            arrows_R[0].animate.set_opacity(0.0),
            run_time=0.6,
        )

        # New resultants after removal
        grads_A_removed = [grads_A[i] for i in range(n) if i != 4]
        grads_B_removed = [grads_B[i] for i in range(n) if i != 0]
        mean_A_new = np.mean(grads_A_removed, axis=0) * arrow_scale
        mean_B_new = np.mean(grads_B_removed, axis=0) * arrow_scale

        res_L_new_end = ball_L_pos + np.array([mean_A_new[0], mean_A_new[1], 0])
        res_R_new_end = ball_R_pos + np.array([mean_B_new[0], mean_B_new[1], 0])

        resultant_L_new = Arrow(
            start=ball_L_pos, end=res_L_new_end,
            color=ORANGE, stroke_width=5, buff=0,
            max_tip_length_to_length_ratio=0.2,
        )
        resultant_R_new = Arrow(
            start=ball_R_pos, end=res_R_new_end,
            color=TEAL, stroke_width=5, buff=0,
            max_tip_length_to_length_ratio=0.2,
        )

        self.play(
            GrowArrow(resultant_L_new),
            GrowArrow(resultant_R_new),
            run_time=0.8,
        )
        self.wait(0.5)

        # ── Angular shifts ──
        def angle_between(v1, v2):
            u1 = v1 / np.linalg.norm(v1)
            u2 = v2 / np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1, 1)))

        angle_L = angle_between(mean_A, mean_A_new)
        angle_R = angle_between(mean_B, mean_B_new)

        angle_L_label = Text(
            f"{angle_L:.0f}° shift", font_size=22, color=ORANGE,
        ).next_to(resultant_L_new.get_end(), DOWN, buff=0.2)
        angle_R_label = Text(
            f"{angle_R:.0f}° shift", font_size=22, color=TEAL,
        ).next_to(resultant_R_new.get_end(), DOWN, buff=0.2)

        verdict_text = Text(
            "Same average — completely different sensitivity",
            font_size=22, color=ACCENT,
        ).to_edge(DOWN, buff=0.4)

        self.play(
            FadeIn(angle_L_label), FadeIn(angle_R_label),
            ReplacementTransform(remove_text, verdict_text),
            run_time=0.6,
        )
        self.wait(2.0)

        # ── Final message ──
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)

        final = VGroup(
            Text("The average hides the disagreement.", font_size=26, color=SOFT_WHITE),
            Text("The disagreement is where influence lives.", font_size=26, color=ACCENT),
        ).arrange(DOWN, buff=0.3).move_to(ORIGIN)

        self.play(FadeIn(final, shift=UP * 0.3), run_time=0.8)
        self.wait(2.5)


class CurvatureAmplifier(ThreeDScene):
    """3D loss surface with a clear knife-edge and plateau. Two equal-magnitude
    gradients on the surface: H^{-1} amplifies the flat-direction one (nothing
    resists) and dampens the steep-direction one (walls push back)."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Title (fixed to screen) ──
        title = Text(
            "Same Force, Different Influence", font_size=34, color=SOFT_WHITE
        )
        subtitle = Text(
            "The landscape decides how far each push carries",
            font_size=20, color=DIM_WHITE,
        )
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(header)
        self.play(FadeIn(header), run_time=0.6)

        # ── Camera ──
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=0.7)

        # ── 3D axes ──
        axes = ThreeDAxes(
            x_range=[-2, 2, 0.5],
            y_range=[-3, 3, 1],
            z_range=[0, 6, 1],
            x_length=4.5,
            y_length=6.5,
            z_length=3.5,
            axis_config={"color": GREY_B, "stroke_width": 1, "include_tip": False},
        )

        # Surface: L = steep * θ₁² + flat * θ₂²
        # Steep in θ₁ (narrow valley walls), flat in θ₂ (wide open floor)
        steep_c = 4.0
        flat_c = 0.25

        surface = Surface(
            lambda u, v: axes.c2p(u, v, steep_c * u ** 2 + flat_c * v ** 2),
            u_range=[-1.4, 1.4],
            v_range=[-2.8, 2.8],
            resolution=(50, 50),
            fill_opacity=0.5,
            stroke_width=0.3,
            stroke_opacity=0.15,
        )
        surface.set_color_by_gradient(BLUE_E, BLUE_C, GREEN_C, YELLOW_C)

        self.play(Create(axes), run_time=0.6)
        self.play(Create(surface), run_time=1.5)

        # ── Rotate camera to show the steep walls ──
        steep_note = Text("Steep walls → high curvature", font_size=18, color=RED_C)
        steep_note.to_corner(DL, buff=0.5)
        self.add_fixed_in_frame_mobjects(steep_note)
        self.play(FadeIn(steep_note), run_time=0.3)
        self.move_camera(theta=-80 * DEGREES, run_time=2.0)
        self.wait(0.5)
        self.play(FadeOut(steep_note), run_time=0.3)

        # ── Rotate to show the flat valley floor ──
        flat_note = Text("Flat floor → low curvature", font_size=18, color=GREEN_C)
        flat_note.to_corner(DL, buff=0.5)
        self.add_fixed_in_frame_mobjects(flat_note)
        self.play(FadeIn(flat_note), run_time=0.3)
        self.move_camera(theta=-10 * DEGREES, run_time=2.0)
        self.wait(0.5)
        self.play(FadeOut(flat_note), run_time=0.3)

        # Return to a good viewing angle
        self.move_camera(theta=-40 * DEGREES, phi=60 * DEGREES, run_time=1.5)

        # ── Ball at bottom of valley ──
        origin = axes.c2p(0, 0, 0)
        ball = Sphere(radius=0.08, color=ACCENT, resolution=(12, 12))
        ball.move_to(origin)
        self.play(FadeIn(ball), run_time=0.3)

        # ── Two gradient arrows of equal length ──
        grad_len = 1.0

        arrow_steep = Arrow3D(
            start=origin,
            end=axes.c2p(grad_len, 0, 0),
            color=RED,
            thickness=0.025,
        )
        arrow_flat = Arrow3D(
            start=origin,
            end=axes.c2p(0, grad_len, 0),
            color=GREEN,
            thickness=0.025,
        )

        g1_lab = MathTex(r"g_1", font_size=24, color=RED)
        g1_lab.move_to(axes.c2p(grad_len + 0.25, 0, 0.15))
        self.add_fixed_orientation_mobjects(g1_lab)

        g2_lab = MathTex(r"g_2", font_size=24, color=GREEN)
        g2_lab.move_to(axes.c2p(0, grad_len + 0.3, 0.15))
        self.add_fixed_orientation_mobjects(g2_lab)

        eq_text = Text("Same magnitude: ‖g₁‖ = ‖g₂‖", font_size=22, color=SOFT_WHITE)
        eq_text.to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(eq_text)

        self.play(
            Create(arrow_steep), Create(arrow_flat),
            FadeIn(g1_lab), FadeIn(g2_lab), FadeIn(eq_text),
            run_time=0.8,
        )
        self.wait(1.2)

        # ── Apply H^{-1} ──
        # H = diag(2*steep_c, 2*flat_c) = diag(8, 0.5)
        # H^{-1} = diag(0.125, 2.0)
        # Steep direction: walls resist → arrow SHRINKS
        # Flat direction: nothing resists → arrow GROWS
        influence_steep = grad_len * 0.25   # visually clear shrink
        influence_flat = grad_len * 2.2     # visually clear growth

        arrow_steep_new = Arrow3D(
            start=origin,
            end=axes.c2p(influence_steep, 0, 0),
            color=RED,
            thickness=0.025,
        )
        arrow_flat_new = Arrow3D(
            start=origin,
            end=axes.c2p(0, influence_flat, 0),
            color=GREEN,
            thickness=0.025,
        )

        g1_new = MathTex(r"H^{-1}g_1", font_size=20, color=RED)
        g1_new.move_to(axes.c2p(influence_steep + 0.35, 0, 0.15))
        self.add_fixed_orientation_mobjects(g1_new)

        g2_new = MathTex(r"H^{-1}g_2", font_size=20, color=GREEN)
        g2_new.move_to(axes.c2p(0, influence_flat + 0.35, 0.15))
        self.add_fixed_orientation_mobjects(g2_new)

        h_text = MathTex(r"\text{Apply } H^{-1}", font_size=26, color=ACCENT)
        h_text.to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(h_text)

        self.play(FadeOut(eq_text), FadeIn(h_text), run_time=0.4)
        self.play(
            FadeOut(arrow_steep), FadeOut(arrow_flat),
            FadeOut(g1_lab), FadeOut(g2_lab),
            FadeIn(arrow_steep_new), FadeIn(arrow_flat_new),
            FadeIn(g1_new), FadeIn(g2_new),
            run_time=1.2,
        )
        self.wait(1.0)

        # ── Result annotation ──
        result = VGroup(
            Text("Steep walls resist the push → barely moves", font_size=18, color=RED_C),
            Text("Flat floor offers no resistance → slides far", font_size=18, color=GREEN_C),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(result)

        self.play(FadeOut(h_text), FadeIn(result), run_time=0.6)
        self.wait(2.0)

        # ── Final ──
        self.play(
            FadeOut(surface), FadeOut(axes), FadeOut(ball),
            FadeOut(arrow_steep_new), FadeOut(arrow_flat_new),
            FadeOut(g1_new), FadeOut(g2_new),
            FadeOut(result), FadeOut(header),
            run_time=0.8,
        )

        final = VGroup(
            Text("It's not about how hard you push.", font_size=26, color=SOFT_WHITE),
            Text("It's about how much the landscape resists.", font_size=26, color=ACCENT),
        ).arrange(DOWN, buff=0.3).move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(final)
        self.play(FadeIn(final, shift=UP * 0.3), run_time=0.8)
        self.wait(2.5)


class FingerprintDetector(Scene):
    """Shows that the batch gradient can be decomposed to identify which
    data points contributed, using inner products as a 'fingerprint detector.'"""

    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("The Fingerprint Detector", font_size=36, color=SOFT_WHITE)
        subtitle = Text(
            "Tracing the batch gradient back to individual data points",
            font_size=20, color=DIM_WHITE,
        )
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.2)
        header.to_edge(UP, buff=0.3)
        self.play(FadeIn(header), run_time=0.6)

        # ── Left side: the batch gradient (what we observe) ──
        left_title = Text("What we observe", font_size=20, color=DIM_WHITE)
        left_title.move_to(LEFT * 4 + UP * 1.5)

        batch_arrow = Arrow(
            start=LEFT * 4.5 + DOWN * 0.5,
            end=LEFT * 3.0 + UP * 0.8,
            color=WHITE, stroke_width=5, buff=0,
            max_tip_length_to_length_ratio=0.15,
        )
        batch_label = MathTex(
            r"\nabla \mathcal{L}", font_size=28, color=WHITE
        ).next_to(batch_arrow.get_end(), UR, buff=0.1)

        self.play(
            FadeIn(left_title),
            GrowArrow(batch_arrow),
            FadeIn(batch_label),
            run_time=0.6,
        )

        # ── Right side: library of known per-sample gradients ──
        right_title = Text("Per-sample gradient library", font_size=20, color=DIM_WHITE)
        right_title.move_to(RIGHT * 2.5 + UP * 1.5)
        self.play(FadeIn(right_title), run_time=0.3)

        n_library = 6
        library_colors = [BLUE, GREEN, ORANGE, PURPLE, RED, TEAL]
        library_names = [f"g_{i+1}" for i in range(n_library)]
        # Similarity scores (inner product with batch gradient)
        scores = [0.12, 0.08, 0.85, 0.05, 0.72, 0.11]

        library_rows = VGroup()
        for i in range(n_library):
            dot = Dot(color=library_colors[i], radius=0.08)
            name = MathTex(library_names[i], font_size=22, color=library_colors[i])
            bar_width = scores[i] * 3.0
            bar = Rectangle(
                width=bar_width, height=0.25,
                color=library_colors[i],
                fill_opacity=0.6,
                stroke_width=0,
            )
            score_text = Text(
                f"{scores[i]:.2f}", font_size=16, color=SOFT_WHITE
            )
            row = VGroup(dot, name).arrange(RIGHT, buff=0.1)
            bar.next_to(row, RIGHT, buff=0.2)
            bar.align_to(row, LEFT).shift(RIGHT * 1.2)
            score_text.next_to(bar, RIGHT, buff=0.1)
            full_row = VGroup(row, bar, score_text)
            library_rows.add(full_row)

        library_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        library_rows.move_to(RIGHT * 2.5 + DOWN * 0.3)

        # Animate: show rows one at a time with bars growing
        for i in range(n_library):
            row_group = library_rows[i]
            # Initially hide the bar and score
            bar = row_group[1]
            score = row_group[2]
            bar.save_state()
            bar.stretch(0, 0, about_edge=LEFT)
            score.set_opacity(0)

            self.play(
                FadeIn(row_group[0]),
                bar.animate.restore(),
                score.animate.set_opacity(1),
                run_time=0.5,
            )

            # Highlight high-score matches
            if scores[i] > 0.5:
                highlight = SurroundingRectangle(
                    row_group, color=ACCENT, stroke_width=2, buff=0.08,
                )
                match_label = Text(
                    "MATCH", font_size=14, color=ACCENT,
                ).next_to(highlight, RIGHT, buff=0.1)
                self.play(
                    Create(highlight), FadeIn(match_label),
                    run_time=0.3,
                )

        self.wait(0.5)

        # ── Connecting arrow: batch gradient → decomposition ──
        connector_text = Text(
            "Inner products reveal who contributed",
            font_size=22, color=ACCENT,
        ).to_edge(DOWN, buff=0.4)

        self.play(FadeIn(connector_text, shift=UP * 0.2), run_time=0.6)
        self.wait(1.5)

        # ── Final message ──
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)

        final = VGroup(
            Text("Every data point leaves a fingerprint.", font_size=26, color=SOFT_WHITE),
            Text("The batch gradient is the crime scene.", font_size=26, color=SOFT_WHITE),
            Text("And these fingerprints don't wash off.", font_size=26, color=ACCENT),
        ).arrange(DOWN, buff=0.3).move_to(ORIGIN)

        self.play(FadeIn(final, shift=UP * 0.3), run_time=0.8)
        self.wait(2.5)
