import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import matplotlib as mpl

colormap = mpl.colormaps["BuPu"]
condition = 'zpdes'
# Create a figure and a single subplot
with open(f'config/summary_radar_v3_{condition}.json', 'r') as f:
    config = json.load(f)
heights = config['wedges']
radius_main = config['main']['height']

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Draw the center circle
center_circle = plt.Circle((0, 0), radius_main, color=colormap(config['main']['color']), fill=True)
ax.add_artist(center_circle)
ax.text(0, 0, "g_score", color='black', fontsize=8,
        ha="center", va="center")
# Define the starting angle for the first segment
start_angle = 0

# Draw the parts and sub-parts
for part, attributes in heights.items():
    main_height = attributes["height"]
    main_color_value = attributes["color"]
    main_color = colormap(main_color_value)
    angle_per_part = 360 / len(heights)

    # Draw the main part
    main_wedge = patches.Wedge(center=(0, 0), r=radius_main + main_height, theta1=start_angle,
                               theta2=start_angle + angle_per_part, width=main_height,
                               color=main_color, alpha=0.5, label=part)
    ax.add_patch(main_wedge)
    midpoint_angle_degrees = start_angle + angle_per_part / 2  # Midpoint angle in degrees
    midpoint_angle_radians = np.radians(midpoint_angle_degrees)  # Convert to radians for np.cos and np.sin
    x_PC = (radius_main + main_height / 2) * np.cos(midpoint_angle_radians)
    y_PC = (radius_main + main_height / 2) * np.sin(midpoint_angle_radians)
    ax.text(x_PC, y_PC, part, color='black', fontsize=8,
            ha="center", va="center")

    # Check if there are children parts to draw
    if "children" in attributes:
        children = attributes["children"]
        sub_start_angle = start_angle
        angle_per_sub_part = angle_per_part / len(children)

        for sub_part, sub_attr in children.items():
            sub_height = sub_attr["height"]
            off_set_observed = 0
            # Retrieve the bar color
            sub_part_color = colormap(1 - sub_height)
            # Draw the sub-part
            sub_wedge = patches.Wedge(center=(0, 0), r=radius_main + main_height + sub_height + off_set_observed,
                                      theta1=sub_start_angle, theta2=sub_start_angle + angle_per_sub_part,
                                      width=sub_height, color=sub_part_color, alpha=1, label=sub_part)
            ax.add_patch(sub_wedge)
            # Check for "bars" or children of the sub-part
            if "children" in sub_attr:
                bar_children = sub_attr["children"]
                bar_width = angle_per_sub_part / len(bar_children)
                bar_start_angle = sub_start_angle

                for bar_part, bar_attr in bar_children.items():
                    bar_height = bar_attr["height"]
                    # Calculate the start and end points of the bar
                    bar_start = radius_main + main_height + sub_height + off_set_observed
                    bar_end = bar_start + bar_height + off_set_observed
                    # Calculate the angle at the midpoint of the bar for correct orientation
                    bar_angle = (bar_start_angle + bar_width / 2) * (np.pi / 180)  # converting to radians
                    # Calculate the start and end points for the bar
                    start_x = bar_start * np.cos(bar_angle)
                    start_y = bar_start * np.sin(bar_angle)
                    end_x = (bar_end - bar_start) * np.cos(bar_angle)
                    end_y = (bar_end - bar_start) * np.sin(bar_angle)
                    # Retrieve the bar color
                    bar_color = colormap(1-bar_height)
                    # Create a line for the bar
                    bar = patches.FancyArrow(start_x, start_y, end_x, end_y,
                                             width=0.1, head_width=0, head_length=0, color=bar_color)
                    ax.add_patch(bar)
                    offset = 0
                    if 'offset' in bar_attr:
                        offset = bar_attr['offset']
                    label_x = (bar_start + bar_height + offset) * np.cos(bar_angle)
                    label_y = (bar_start + bar_height + offset) * np.sin(bar_angle)
                    # Adjust alignment based on the angle quadrant
                    if 0 <= bar_angle <= np.pi / 2:  # Right half of the circle
                        ha, va = 'left', 'bottom'
                    elif np.pi / 2 < bar_angle <= 3 * np.pi / 2:
                        ha, va = 'right', 'bottom'
                    else:  # Left half of the circle
                        ha, va = 'center', 'center'
                    ha, va = 'center', 'center'
                    # Rotate the text by 180 degrees for the top half of the circle
                    rotation_angle = np.degrees(bar_angle)
                    if np.pi / 2 < bar_angle < 3 * np.pi / 2:  # Top half of the circle
                        rotation_angle += 180
                    ax.text(label_x, label_y, bar_part, color='black', fontsize=8,
                            ha=ha, va=va, rotation=rotation_angle)
                    bar_start_angle += bar_width

            sub_start_angle += angle_per_sub_part

    start_angle += angle_per_part

# Set the aspect of the plot to be equal
ax.set_aspect('equal')

# Set limits to ensure the entire figure is visible
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# Remove axes for a cleaner look
ax.axis('off')
# Create a ScalarMappable and initialize with your colormap and the range of your data
sm = mpl.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))

# Important if you're using this in a loop or function, otherwise it might be unnecessary
sm.set_array([])

# Create the colorbar
plt.colorbar(sm, fraction=0.025)
# Display the plot
fig.savefig(f'summary_radar_{condition}.png', transparent=True)
# plt.show()
