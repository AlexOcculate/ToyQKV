from manim import *
import numpy as np


class AttentionPull3DScene(ThreeDScene):
    def construct(self):
        # 1. SETUP GLOBAL CAMERA & WORLD OBJECTS
        self.set_camera_orientation(phi=65 * DEGREES, theta=-60 * DEGREES, zoom=0.85)
        self.add(
            ThreeDAxes(
                x_range=[-4, 4, 1],
                y_range=[-4, 4, 1],
                z_range=[-4, 4, 1],
                x_length=6,
                y_length=6,
                z_length=6,
                axis_config={"stroke_width": 1, "color": GRAY_E},
            )
        )

        # Screen Fixed Title & Muted Axis Labels
        self.add_fixed_in_frame_mobjects(
            Tex("3D Multi-Angle Attention Pull", color=WHITE).scale(1.2).to_corner(UL)
        )
        self.add_fixed_orientation_mobjects(
            Tex("X", color=GRAY_D).scale(0.5).move_to(RIGHT * 3.25),
            Tex("Y", color=GRAY_D).scale(0.5).move_to(UP * 3.25),
            Tex("Z", color=GRAY_D).scale(0.5).move_to(OUT * 3.25),
        )

        # 2. DATA CLUSTERS & CALCULATION TARGET
        token_data = {
            "river": {"pos": np.array([-2.5, -2.0, -1.5]), "color": BLUE_D},
            "bank": {"pos": np.array([2.5, 2.5, 2.0]), "color": GOLD_D},
            "muddy": {"pos": np.array([-1.5, 2.0, -2.5]), "color": RED_D},
        }
        orig_bank_pos = token_data["bank"]["pos"]
        target_bank_pos = (
            (0.75 * token_data["river"]["pos"])
            + (0.15 * token_data["muddy"]["pos"])
            + (0.1 * orig_bank_pos)
        )

        # 3. COMPACT RE-MAPPED ORTHOGRAPHIC PANEL SUB-DECK SYSTEMS
        # FIX: Explicit index unpacking ensures consistent flat array dimensions [Scalar, Scalar, Scalar]
        def get_projected_points(v):
            x_scaled = v[0] * 0.35
            y_scaled = v[1] * 0.35
            z_scaled = v[2] * 0.35
            return (
                np.array([x_scaled, y_scaled, -4.5]),
                np.array([x_scaled, 4.5, z_scaled]),
                np.array([-4.5, y_scaled, z_scaled]),
            )

        panels = [
            (
                IN,
                -4.5,
                "XY View (Top-Down)",
                PURPLE_E,
                lambda p: p.move_to([0, 0, -4.5]),
                lambda l, p: l.next_to(p, UP, buff=0.1),
            ),
            (
                UP,
                4.5,
                "XZ View (Frontal)",
                TEAL_E,
                lambda p: p.rotate(90 * DEGREES, axis=RIGHT).move_to([0, 4.5, 0]),
                lambda l, p: l.move_to([0, 4.5, 1.6]),
            ),
            (
                RIGHT,
                -4.5,
                "YZ View (Profile)",
                "#5c2e00",
                lambda p: p.rotate(90 * DEGREES, axis=UP).move_to([-4.5, 0, 0]),
                lambda l, p: l.move_to([-4.5, 0, 1.6]),
            ),
        ]

        lbl_group = VGroup()
        for norm, offset, name, col, space_func, lbl_func in panels:
            plane = NumberPlane(
                x_range=[-4, 4, 1],
                y_range=[-4, 4, 1],
                axis_config={
                    "stroke_width": 1,
                    "color": GRAY_E,
                    "include_numbers": False,
                },
                background_line_style={
                    "stroke_color": col,
                    "stroke_width": 1,
                    "stroke_opacity": 0.2,
                },
            ).scale(0.35)
            space_func(plane)
            lbl = Tex(name, color=GRAY_D).scale(0.5)
            lbl_func(lbl, plane)
            self.add(plane)
            lbl_group.add(lbl)
        self.add_fixed_orientation_mobjects(*lbl_group)

        # 4. INSTANTIATE PRIMARY MOLECULES & ATOMS
        dots, shadows = VGroup(), VGroup()
        bank_refs = {}

        for name, info in token_data.items():
            d = Dot3D(point=info["pos"], color=info["color"], radius=0.12)
            xy, xz, yz = get_projected_points(info["pos"])
            s_xy = Dot(xy, color=info["color"], radius=0.06)
            s_xz = Dot(xz, color=info["color"], radius=0.06)
            s_yz = Dot(yz, color=info["color"], radius=0.06)
            dots.add(d)
            shadows.add(s_xy, s_xz, s_yz)

            if name == "bank":
                bank_refs.update({"dot": d, "xy": s_xy, "xz": s_xz, "yz": s_yz})

        self.add(dots, shadows)

        # Flat Dynamic Billboards Setup (Add explicit role markers for clarity)
        river_lbl = (
            Tex("river", color=BLUE_D)
            .scale(0.75)
            .move_to(token_data["river"]["pos"] + UP * 0.3 + RIGHT * 0.3)
        )
        muddy_lbl = (
            Tex("muddy", color=RED_D)
            .scale(0.75)
            .move_to(token_data["muddy"]["pos"] + UP * 0.3 + RIGHT * 0.3)
        )

        # Origin static billboard label
        bank_orig_lbl = (
            Tex("bank", color=GOLD_D)
            .scale(0.75)
            .move_to(orig_bank_pos + UP * 0.3 + RIGHT * 0.3)
        )
        # Moving tracking billboard label instance
        bank_curr_lbl = (
            Tex("bank", color=GOLD_A)
            .scale(0.75)
            .move_to(orig_bank_pos + UP * 0.3 + RIGHT * 0.3)
        )

        self.add_fixed_orientation_mobjects(
            river_lbl, muddy_lbl, bank_orig_lbl, bank_curr_lbl
        )
        self.wait(1)

        # 5. ANIMATE VECTOR WARP & LIVE UPDATER HOOKS
        pull_vector = Arrow3D(
            start=orig_bank_pos,
            end=target_bank_pos,
            color=GOLD_A,
            thickness=0.005,  # @#$%
        )
        self.play(Create(pull_vector), run_time=1.5)
        self.wait(0.5)

        # Drop a static reference clone dot at the origin before traveling
        static_bank_origin_dot = Dot3D(point=orig_bank_pos, color=GOLD_D, radius=0.12)
        self.add(static_bank_origin_dot)

        def sync_updater(mob):
            curr = mob.get_center()
            xy, xz, yz = get_projected_points(curr)
            bank_refs["xy"].move_to(xy)
            bank_refs["xz"].move_to(xz)
            bank_refs["yz"].move_to(yz)
            bank_curr_lbl.move_to(curr + UP * 0.3 + RIGHT * 0.3)

        bank_refs["dot"].add_updater(sync_updater)

        # Animate transit along the attention pull vector field track
        self.play(
            bank_refs["dot"].animate.move_to(target_bank_pos),
            FadeOut(pull_vector),
            run_time=3.5,
            rate_func=smooth,
        )
        bank_refs["dot"].remove_updater(sync_updater)

        # Update text string layout smoothly upon reaching destination cell context
        final_context_lbl = (
            Tex("bank (context)", color=GOLD_A)
            .scale(0.75)
            .move_to(target_bank_pos + UP * 0.3 + RIGHT * 0.3)
        )
        self.remove(bank_curr_lbl)
        self.add_fixed_orientation_mobjects(final_context_lbl)
        self.wait(1)

        # 6. CAMERA GLIDE
        self.move_camera(phi=45 * DEGREES, theta=35 * DEGREES, zoom=0.9, run_time=3.0)
        self.wait(2)
