from manim import *
import numpy as np


class AttentionPull3DScene(ThreeDScene):
    def construct(self):
        # ---------------------------------------------------------
        # 1. SETUP GLOBAL 3D WORLD & CAMERA CONFIG
        # ---------------------------------------------------------
        self.set_camera_orientation(phi=65 * DEGREES, theta=-60 * DEGREES, zoom=0.85)

        # 3D Main Axes
        main_axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            z_length=6,
            axis_config={"stroke_width": 2, "color": GRAY_C},
        )
        self.add(main_axes)

        # Title Label in Screen Space (Fixed position on screen)
        title = Tex("3D Multi-Angle Attention Pull", color=WHITE)
        title.scale(1.2)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)

        # ---------------------------------------------------------
        # 2. DEFINE THE EMBEDDING DATA POINTS (Spatially Spread)
        # ---------------------------------------------------------
        token_data = {
            "river": {"pos": np.array([-2.5, -2.0, -1.5]), "color": BLUE_D},
            "bank": {"pos": np.array([2.5, 2.5, 2.0]), "color": GOLD_D},
            "muddy": {"pos": np.array([-1.5, 2.0, -2.5]), "color": RED_D},
        }

        river_pos = token_data["river"]["pos"]
        muddy_pos = token_data["muddy"]["pos"]

        # Calculate context-dependent destination vector
        target_bank_pos = (
            (0.75 * river_pos) + (0.15 * muddy_pos) + (0.1 * token_data["bank"]["pos"])
        )

        # ---------------------------------------------------------
        # 3. CONSTRUCT PERSPECTIVE PROJECTION MULTI-PANELS
        # ---------------------------------------------------------
        def create_projection_plane(normal, offset_distance, label_text, color):
            plane = NumberPlane(
                x_range=[-4, 4, 1],
                y_range=[-4, 4, 1],
                background_line_style={
                    "stroke_color": color,
                    "stroke_width": 1,
                    "stroke_opacity": 0.25,
                },
            ).scale(0.35)

            if np.array_equal(normal, IN):  # XY Plane
                plane.move_to(np.array([0, 0, offset_distance]))
            elif np.array_equal(normal, RIGHT):  # YZ Plane
                plane.rotate(90 * DEGREES, axis=UP)
                plane.move_to(np.array([offset_distance, 0, 0]))
            elif np.array_equal(normal, UP):  # XZ Plane
                plane.rotate(90 * DEGREES, axis=RIGHT)
                plane.move_to(np.array([0, offset_distance, 0]))

            lbl = Tex(label_text, color=color).scale(0.5).next_to(plane, UP, buff=0.1)
            return VGroup(plane, lbl)

        # Position tracking decks safely at background limits
        xy_panel = create_projection_plane(IN, -4.5, "XY View (Top-Down)", PURPLE_A)
        xz_panel = create_projection_plane(UP, 4.5, "XZ View (Frontal)", TEAL_A)
        yz_panel = create_projection_plane(RIGHT, -4.5, "YZ View (Profile)", ORANGE)

        self.add(xy_panel, xz_panel, yz_panel)

        # ---------------------------------------------------------
        # 4. INSTANTIATE PLACEMENT DOTS & BILLBOARD LABELS
        # ---------------------------------------------------------
        dots = VGroup()
        shadow_dots = VGroup()

        # FIX: Map scalar coordinate positions component-by-component
        # to ensure homogenous array shapes [X, Y, Z]
        def get_projected_points(vec3d):
            x_scaled = vec3d[0] * 0.35
            y_scaled = vec3d[1] * 0.35
            z_scaled = vec3d[2] * 0.35

            xy_pt = np.array([x_scaled, y_scaled, -4.5])
            xz_pt = np.array([x_scaled, 4.5, z_scaled])
            yz_pt = np.array([-4.5, y_scaled, z_scaled])
            return xy_pt, xz_pt, yz_pt

        # Trace variables for "bank" dynamic updating
        bank_dot = None
        bank_xy_s = None
        bank_xz_s = None
        bank_yz_s = None

        for name, info in token_data.items():
            pt = info["pos"]
            col = info["color"]

            # Parent 3D nodes
            d = Dot3D(point=pt, color=col, radius=0.12)
            dots.add(d)

            # Projection markers
            xy_pt, xz_pt, yz_pt = get_projected_points(pt)
            s_xy = Dot(point=xy_pt, color=col, radius=0.06)
            s_xz = Dot(point=xz_pt, color=col, radius=0.06)
            s_yz = Dot(point=yz_pt, color=col, radius=0.06)
            shadow_dots.add(s_xy, s_xz, s_yz)

            if name == "bank":
                bank_dot = d
                bank_xy_s = s_xy
                bank_xz_s = s_xz
                bank_yz_s = s_yz

        self.add(dots, shadow_dots)

        # Native dynamic billboard labels (always look straight at the lens)
        river_lbl = (
            Tex("river", color=BLUE_D)
            .scale(0.75)
            .move_to(token_data["river"]["pos"] + UP * 0.3 + RIGHT * 0.3)
        )
        bank_lbl = (
            Tex("bank", color=GOLD_D)
            .scale(0.75)
            .move_to(token_data["bank"]["pos"] + UP * 0.3 + RIGHT * 0.3)
        )
        muddy_lbl = (
            Tex("muddy", color=RED_D)
            .scale(0.75)
            .move_to(token_data["muddy"]["pos"] + UP * 0.3 + RIGHT * 0.3)
        )

        self.add_fixed_orientation_mobjects(river_lbl, bank_lbl, muddy_lbl)

        self.wait(1)

        # ---------------------------------------------------------
        # 5. EXECUTE THE ATTENTION VECTOR WARP & PLANE SYNCHRONIZATION
        # ---------------------------------------------------------
        pull_vector = Arrow3D(
            start=token_data["bank"]["pos"],
            end=target_bank_pos,
            color=GOLD_A,
            thickness=0.02,
        )

        self.play(Create(pull_vector), run_time=1.5)
        self.wait(0.5)

        # Explicit projection updater function to trace movements across all sub-decks safely
        def update_projections(mob):
            curr_pos = mob.get_center()
            xy_pt, xz_pt, yz_pt = get_projected_points(curr_pos)
            bank_xy_s.move_to(xy_pt)
            bank_xz_s.move_to(xz_pt)
            bank_yz_s.move_to(yz_pt)
            bank_lbl.move_to(curr_pos + UP * 0.3 + RIGHT * 0.3)

        bank_dot.add_updater(update_projections)

        # Synchronised spatial animation warp
        self.play(
            bank_dot.animate.move_to(target_bank_pos),
            FadeOut(pull_vector),
            run_time=3.5,
            rate_func=smooth,
        )

        bank_dot.remove_updater(update_projections)
        self.wait(1)

        # ---------------------------------------------------------
        # 6. DYNAMIC CAMERA SWEEP
        # ---------------------------------------------------------
        self.move_camera(phi=45 * DEGREES, theta=35 * DEGREES, zoom=0.9, run_time=3.0)
        self.wait(2)
