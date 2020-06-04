import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from PIL import Image
import io


def alpha_scaling(x, c, s):
    """
    Function to compute alpha values based on data magnitude.
    """
    return 1 - (1 - np.exp(-c * s) + np.exp((x - c) * s)) ** -1


class DataLayer:
    """
    Convenience class for plotting projected lon/lat data.
    """
    def __init__(self, data):

        self.__data = data
        self.__vmin = None
        self.__vmax = None

    def draw(self, ax, vmin, vmax, target_projection, target_resolution=(1000, 1000), c=0.5, s=1, cmap='RdBu'):
        """
        Draws the projected data set in a given Axes instance.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axes instance
        vmin : float
            lower data limit
        vmax : float
            upper data limit
        target_projection : cartopy.crs.Projection
            projection applied to the lon/lat data
        target_resolution : tuple
            resolution of the output grid
        c : float
            offset parameter for alpha computation
        s : float
            slope parameter for alpha computation
        cmap : str
            matplotlib colormap
        """
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        data, extent = warp_array(self.__data, target_projection, ccrs.PlateCarree(), target_res=target_resolution)

        values_normalized = (data - vmin) / (vmax - vmin)
        magnitude = np.abs(data / vmax)
        magnitude[magnitude > 1] = 1
        magnitude[data.mask] = 0

        alpha = Image.fromarray(np.uint8(alpha_scaling(magnitude, c, s) * 255))
        bitmap = Image.fromarray(np.uint8(cmap(values_normalized) * 255))
        bitmap.putalpha(alpha)

        return ax.imshow(bitmap, extent=extent)


class BlueMarble:
    """
    Convenience class for handling NASA Blue Marble images in map projections.
    """
    def __init__(self, file_name):

        img = Image.open(file_name)
        self.__data = np.array(img)

    def draw(self, ax, target_projection, target_resolution=(1000, 1000)):
        """
        Draws the projected image in a given Axes instance.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axes instance
        target_projection : cartopy.crs.Projection
            projection applied to the lon/lat data
        target_resolution : tuple
            resolution of the output grid

        """
        data, extent = warp_array(self.__data, target_projection, ccrs.PlateCarree(), target_res=target_resolution)

        img = Image.fromarray(data)
        alpha = Image.fromarray(np.uint8(~data.mask[:, :, 0]*255))
        img.putalpha(alpha)

        return ax.imshow(img, extent=extent)


class OrthographicAtmosphere:
    """
    Simulates atmospheric glow for global orthographic projections.
    """
    def __init__(self, atmosphere_height, color, radius_earth=6378136.3):

        self.radius = radius_earth + atmosphere_height
        self.radius_earth = radius_earth
        self.color = color

    def draw(self, ax):
        """
        Draws a circular atmosphere in a given Axes instance.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axes instance
        """
        patch_count = 100
        radii = np.linspace(self.radius, self.radius_earth, patch_count)
        alpha_values = np.linspace(0.5/patch_count, 1/patch_count, radii.size)

        circles = []
        for k in range(patch_count):

            circle = mpl.patches.Circle((0, 0), radius=radii[k], alpha=alpha_values[k], fill=True,
                                       fc=self.color, edgecolor=None, linewidth=0, zorder=-10000+k)
            ax.add_patch(circle)
            circles.append(circle)

        return circles


class PanelMetaData:
    """
    Container for panel metadata.
    """
    def __init__(self, data_set, vmin, vmax, title, label, cmap, lon0, lat0):

        self.data_set = data_set
        self.vmin = vmin
        self.vmax = vmax
        self.title = title
        self.label = label
        self.cmap = cmap
        self.projection1 = ccrs.Orthographic(central_latitude=-lat0, central_longitude=lon0)
        self.projection2 = ccrs.Orthographic(central_latitude=lat0, central_longitude=lon0 if lon0 == 0 else lon0+180)


trend = np.load('data/GOCO06s_trend_density.npy')
amplitude = np.load('data/GOCO06s_annualAmplitude_water_height.npy')
anomalies = np.load('data/GOCO06s_static_anomalies.npy')

blue_marble = BlueMarble('data/bm_lowres.png')
atmosphere = OrthographicAtmosphere(750e3, 'w')

panels = [PanelMetaData(trend, -300, 300, 'a) long term trend', 'mass change [kg m$^{-2}$ year$^{-1}$]', 'RdBu', 0, 90),
          PanelMetaData(amplitude, 0, 35, 'b) annual amplitude', 'water height [cm]', 'Blues', -70, -20),
          PanelMetaData(anomalies, -100, 100, 'c) static gravity field', 'gravity anomalies [mgal]', 'PiYG', 135, 35)]

text_color = 'w'
font_size = 10
plt.rcParams.update({'font.family': 'arial', 'font.weight': 'bold',
                     'font.size': font_size, 'text.color': text_color,
                     'axes.labelcolor': text_color, 'xtick.color': text_color,
                     'ytick.color': text_color})

target_resolution = (2000, 2000)

panel_bitmaps = []
for k, panel in enumerate(panels):

    fig = plt.figure(figsize=(5.25, 5.25))

    ax1 = plt.subplot(1, 1, 1)

    ax1.set_title(panel.title)

    atmosphere.draw(ax1)
    blue_marble.draw(ax1, panel.projection1, target_resolution=target_resolution)

    data_layer = DataLayer(panel.data_set)
    data_layer.draw(ax1, panel.vmin, panel.vmax, panel.projection1, c=0.4, s=10,
                    cmap=panel.cmap, target_resolution=target_resolution)

    ax1.set_xlim((-atmosphere.radius, atmosphere.radius))
    ax1.set_ylim((-atmosphere.radius, atmosphere.radius))

    ax1.set_axis_off()

    ax1_bbox = ax1.get_position()
    p1 = ax1_bbox.get_points()[0]
    p2 = ax1_bbox.get_points()[1]

    ax1_width = p2[0] - p1[0]
    ax1_height = p2[1] - p1[1]

    colorbar_width = 0.75

    ax2 = fig.add_axes([p1[0]+ax1_width*(1-colorbar_width)*0.5, p1[1]-0.01, ax1_width*colorbar_width, 0.02])
    norm = mpl.colors.Normalize(vmin=panel.vmin, vmax=panel.vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=plt.get_cmap(panel.cmap), norm=norm, orientation='horizontal',
                                    extend='max' if panel.vmin == 0 else 'both', extendfrac=0.1)

    cb1.set_label(panel.label)

    ax3 = fig.add_axes([p1[0], p1[1]-ax1_height-0.075, ax1_width, ax1_height])

    atmosphere.draw(ax3)
    blue_marble.draw(ax3, panel.projection2, target_resolution=target_resolution)
    data_layer.draw(ax3, panel.vmin, panel.vmax, panel.projection2, c=0.4, s=10,
                    cmap=panel.cmap, target_resolution=target_resolution)

    ax3.set_xlim((-atmosphere.radius, atmosphere.radius))
    ax3.set_ylim((-atmosphere.radius, atmosphere.radius))

    ax3.set_axis_off()

    inmemory_file = io.BytesIO()
    plt.savefig(inmemory_file, dpi=300, bbox_inches='tight', facecolor='k')
    inmemory_file.seek(0)
    panel_bitmaps.append(Image.open(inmemory_file))

buffer = 50

total_width = 0
total_height = 0
for image in panel_bitmaps:
    width, height = image.size

    total_width += width
    total_height = max(total_height, height)

total_width += (len(panel_bitmaps)-1)*buffer
canvas = Image.new('RGB', (total_width, total_height), (0, 0, 0))

offsetx = 0
for image in panel_bitmaps:
    width, height = image.size

    canvas.paste(image, (offsetx, 0))
    offsetx += width + buffer

canvas.save('GOCO06s_mosaic.png')