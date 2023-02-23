"""
A GRAPHICAL USER INTERFACE TO ANNOTATE ONE QUERY
"""

# python's native libraries
from typing import List

# matplotlib to implement the interface
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import RadioButtons, Button, CheckButtons, TextBox

# custom module
from tools.utils import get_cycles, SelectedMarker
from core.memory import Query

# suppress warning
import warnings
warnings.filterwarnings("ignore")

# remove the toolbar
mpl.rcParams['toolbar'] = 'None'
mpl.rcParams['keymap.pan'] = []
mpl.rcParams['keymap.zoom'] = []
mpl.rcParams['keymap.fullscreen'] = []
mpl.rcParams['keymap.quit'] = []
mpl.rcParams['keymap.save'] = []
mpl.rcParams['keymap.yscale'] = []

# array of colors to use
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey',
          'tab:olive', 'tab:cyan']

# highlight color
highlight = (1, 1, 0.5, 0.5)


def _plot_by_point(ax, data, y, atts):
    """
    Plot the data on a point-based axis

    Parameters
    ----------
    ax
    data
    y
    atts

    Returns
    -------

    """
    idx_cycles = get_cycles(y)
    ax.plot(data[atts], alpha=0.3, c='k', linestyle=':')
    for _, idx in enumerate(idx_cycles):
        ax.plot(data.loc[idx, atts], c=colors[_ % len(colors)])


def _plot_by_time(ax, data, y, atts):
    """
    Plot the data on a time-based axis
    """
    idx_cycles = get_cycles(y)
    ax.plot(data['time'], data, alpha=0.3, c='k', linestyle=':')
    for _, idx in enumerate(idx_cycles):
        ax.plot(data.loc[idx, 'time'], data.loc[idx, atts], c=colors[_ % len(colors)])


class ChangeAxis:
    """Event handler for the buttons to change plotting axis"""

    def __init__(self, fig, axes, atts, ext_names):
        self.fig = fig
        self.axes = axes
        self.atts = atts
        self.ext_names = ext_names
        self.n_exts = len(self.ext_names)

        self.y = None
        self.df = None
        self.display_mode = 0       # 0 = display by point, 1 = display by time
        self.select_mode = 0        # 0 = whole segmentation, 1 = individual cycles

    def set_data(self, df, y):
        """Set the inner data"""
        self.df = df
        self.y = y

    def change_axes(self, label):
        """Change the plotting x-axes based on the clicked radio button"""
        if label == 'By time':
            if self.display_mode != 1:  # avoid the case when user clicks the same option twice
                self.display_mode = 1
                self.show_time()
        else:
            if self.display_mode != 0:
                self.display_mode = 0
                self.show_point()

    def show_point(self):
        """Change the axes to point-based"""
        for i in range(self.n_exts):
            self._clear_axes(self.axes[i])
            self.axes[i].set_ylabel(self.ext_names[i])
            _plot_by_point(ax=self.axes[i], data=self.df, y=self.y[self.ext_names[i]], atts=self.atts)

        self.axes[0].set_title('DOOR')
        self._redraw()

    def show_time(self):
        """Change the axes to time-based"""
        for i in range(self.n_exts):
            self._clear_axes(self.axes[i])
            self.axes[i].set_ylabel(self.ext_names[i])
            _plot_by_time(ax=self.axes[i], data=self.df, y=self.y[self.ext_names[i]], atts=self.atts)

        self.axes[0].set_title('DOOR')
        self._redraw()

    def _clear_axes(self, ax):
        """Clear the axes but keep the color (selected result)"""
        regions = [child for child in ax.get_children() if type(child) is Polygon]
        ax.clear()
        for r in regions:
            xmin = r.get_xy()[0][0]
            xmax = r.get_xy()[2][0]
            # if now it's 0 (point-based) then before it was time-based (1) --> change from timestamp to point
            if self.display_mode == 0:
                xmin = self.df.index[self.df.iloc[:, 0] == xmin].values[0]
                xmax = self.df.index[self.df.iloc[:, 0] == xmax].values[0]
            # if now it's 1 (time-based) then before it was point-based (0) --> change from point to timestap
            else:
                xmin = self.df.iloc[int(xmin), 0]
                xmax = self.df.iloc[int(xmax), 0]
            ax.axvspan(xmin=xmin, xmax=xmax, facecolor=highlight, linestyle='--', linewidth=1.5, edgecolor='k')

    def _redraw(self):
        """Redraws the figure and axes"""
        self.fig.canvas.draw()


class GUIAnnotation:
    """
    Build the GUI to allow the user to annotate a query
    """

    def __init__(self, n_exts, ext_names, atts):
        """
        Initialize the annotation interface
        """
        # inner data structure
        self.n_exts = n_exts            # number of extractors
        self.ext_names = ext_names      # name of each extractor
        self.atts = atts                # list of attributes to display

        # new data being processed
        self.current_query = None       # the query that is currently being processed
        self.selected = []              # the selected cycles and their label (SelectedMarker)
        self.data = None                # the current data being shown to the uer
        self.y = None                   # the flattened array of markers currently shown to the user

        # GUI-related
        self.ncols = 1                  # number of columns (2 = if both door and footstep, 1 if door only)
        self.fig, self.ax = None, None  # figure and axes of matplotlib
        self.ax_ext_map = {}            # mao an axis to an extractor
        self.is_confirmed = False       # whether a cycle label is confirmed
        self.select_mode = 0            # 0 = whole segmentation, 1 = separate cycle
        self.rb_select_mode = None      # radio button (whole or individual cycle selection)
        self.rb_xaxes = None            # radio button (plot by point or by time)
        self.btn_refresh = None         # button to clear the selection

        # naming cycles
        self.naming_opt_labels = ['Label cycles']   # a list for the radio button to work
        # self.txtbox_names = []          # the textbox where the cycle label is inputted
        self.is_naming = False          # False = allows selecting/deleting, True = only allow naming
        self.cbx_naming = None          # button to name the cycle
        self.txtbox = None              # textbox to name the cycle
        self.lb_naming_selected = None  # the label to show which cycle is being selected
        self.curr_idx_cyc = None        # the index of the marker being selected for labeling

    def _reset(self):
        """
        Reset all inner data

        Returns
        -------

        """
        # inner data
        self.current_query = None
        self.data = None
        self.selected = []
        self.y = None

        # gui-related setting
        self.select_mode = 0
        self.is_confirmed = False
        self.is_naming = False

    def show_new_query(self, query: Query) -> List:
        """
        Updates the plots in the GUI with new data
        """
        # reset the GUI
        self._reset()

        # get the inner data of a query
        fname = query.query_id
        self.data = query.data
        self.y = {k: v.flattened for k, v in query.candidates.items()}
        self.current_query = query

        # create the components, set data in the change axis handler
        self._create_components()
        self.ca_handler.set_data(self.data, self.y)

        # init the plots
        for i in range(self.n_exts):
            self.ax[i].set_ylabel(self.ext_names[i])
            _plot_by_point(ax=self.ax[i], data=self.data, y=self.y[self.ext_names[i]], atts=self.atts)

        # reset the confirmation state
        self.is_confirmed = False

        # mouse press event handler to let users select the cycles
        self.fig.canvas.mpl_connect('button_press_event', self._cycle_manipulation)

        # key press (Enter) to confirm choices
        self.fig.canvas.mpl_connect('key_press_event', self._keypress)

        # button event to clear all cycles
        self.btn_refresh.on_clicked(self._clear_all_selection)

        # miscellaneous decoration
        self.ax[0].set_title('DOOR')
        plt.tight_layout(rect=(0.03, 0.06, 0.73, 0.95))
        plt.suptitle(fname, fontweight=600, fontsize=12)
        plt.show()

        # return the selected items (deduplicated)
        return list(set(self.selected))

    def _create_components(self):
        """
        Create the necessary components of the GUI

        Returns
        -------

        """
        # create the figure and axes
        self.fig, self.ax = plt.subplots(nrows=self.n_exts, ncols=self.ncols, figsize=(10, self.n_exts*2.7), num='gui')

        # name each segmentation by extractor
        for i in range(len(self.ext_names)):
            self.ax[i].set_ylabel(self.ext_names[i])
            self.ax_ext_map[i] = self.ext_names[i]

        # radio buttons to change the scale of the x-axes ([x, y, width, height])
        self.rb_xaxes = RadioButtons(plt.axes([0.75, 0.80, 0.22, 0.1]), ('By point', 'By time'))
        for circ in self.rb_xaxes.circles:
            circ.height *= 2
            circ.width *= 0.8
        self.ca_handler = ChangeAxis(fig=self.fig, axes=self.ax, atts=self.atts, ext_names=self.ext_names)
        self.ca_handler.display_mode = 0
        self.rb_xaxes.on_clicked(self.ca_handler.change_axes)

        # radio buttons to change the selection mode
        self.rb_select_mode = RadioButtons(plt.axes([0.75, 0.65, 0.22, 0.1]),
                                           ('Entire segmentation', 'Individual cycles'))
        self.ca_handler.select_mode = self.select_mode
        for circ in self.rb_select_mode.circles:
            circ.height *= 2
            circ.width *= 0.8
        self.rb_select_mode.on_clicked(lambda label: self._change_select_mode(label))

        # button to clear selected cycles
        self.btn_refresh = Button(plt.axes([0.75, 0.55, 0.22, 0.05]), 'Clear all')

        # button to name the cycles
        self.cbx_naming = CheckButtons(plt.axes([0.75, 0.45, 0.22, 0.05]), self.naming_opt_labels)
        for i_opt in range(len(self.cbx_naming.rectangles)):
            rect = self.cbx_naming.rectangles[i_opt]
            rect.set_height(0.4)
            rect.set_width(0.08)
            check = self.cbx_naming.lines[i_opt]
            check[0].set_data([rect.get_x(), rect.get_x() + rect.get_width()],
                              [rect.get_y() + rect.get_height(), rect.get_y()])
            check[1].set_data([rect.get_x(), rect.get_x() + rect.get_width()],
                              [rect.get_y(), rect.get_y() + rect.get_height()])
        self.cbx_naming.on_clicked(self._turn_on_off_naming)

        # textbox to enter the cycle label
        self.lb_naming_selected = TextBox(ax=plt.axes([0.75, 0.35, 0.22, 0.05]), label='')
        self.lb_naming_selected.set_val('Cycle labeling disabled')
        self.lb_naming_selected.active = False
        self.lb_naming_selected.color = 'w'
        self.lb_naming_selected.hovercolor = 'w'
        self.txtbox = TextBox(ax=plt.axes([0.75, 0.25, 0.22, 0.05]), label='')
        self.txtbox.active = False
        self.txtbox.on_submit(self._submit_cycle_label)

        # reset all textboxes
        # self.txtbox_names.clear()
        self.is_naming = False

        # add text
        # self.queue_text = plt.text(x=0.05, y=0.03, s='No file in queue', fontsize=12, transform=self.fig.transFigure)
        self.ax[0].set_title('DOOR')

    #region Button event
    def _keypress(self, event):
        """Press enter to confirm choices and pass to the next query"""
        # TODO or we can add a button so as to make the app more intuitive
        if event.key == ' ':
            self.is_confirmed = True
            plt.close()

    def _clear_all_selection(self, event):
        """
        Clear all the selected cycles

        Returns
        -------

        """
        # clear the list of selected cycles
        self.selected = []

        # refresh the axes
        for i in range(self.n_exts):
            self.ax[i].clear()
            if self.ca_handler.display_mode == 0:
                _plot_by_point(ax=self.ax[i], data=self.data, atts=self.atts, y=self.y[self.ext_names[i]])
            else:
                _plot_by_time(ax=self.ax[i], data=self.data, atts=self.atts, y=self.y[self.ext_names[i]])
            self.ax[i].set_ylabel(self.ext_names[i])
        self.ax[0].set_title('DOOR')

        # remove all the textboxes (for cycle labeling)
        # for tb in self.txtbox_names:
        #     tb.ax.remove()
        # self.txtbox_names.clear()

        # finally, redraw everything
        self.fig.canvas.draw()
    #endregion

    #region Cycle manipulation events (select, unselect)
    def _cycle_manipulation(self, event):
        """
        When user clicks in the plots to select/unselect cycles,
        or to give name of a selected cycle (if naming mode is on)

        Parameters
        ----------
        event

        Returns
        -------

        """
        if event.inaxes is not None:
            ax = event.inaxes
            if self.is_naming:              # left click when naming is on --> name a cycle
                self._name_cycles(event=event, ax=ax)
            else:
                if event.button == 3:       # right click --> un-select the cycles
                    self._cycle_unselection(event=event, ax=ax)
                elif event.button == 1:     # left click --> select a cycle
                    self._cycle_selection(event=event, ax=ax)
            self.fig.canvas.draw()

    def _name_cycles(self, event, ax):
        """
        Label a cycle

        Parameters
        ----------
        event
        ax

        Returns
        -------

        """
        # the ID of the axis being chosen
        idx_ax = self.fig.axes.index(ax)

        # if an axis that contains data is selected
        if idx_ax in self.ax_ext_map:
            # the selected extractor
            ext = self.ax_ext_map[idx_ax]

            # find the cycle we are to give labels to
            indices, _ = self._get_cycle_indices_on_click(x=event.xdata, y=self.y[ext])
            if indices is None:
                return
            #  retrieve the selected marker
            ext_markers = self.current_query.candidates[ext].markers
            mid = [mk.mid for mk in ext_markers if mk.mstart >= indices[0] and mk.mend <= indices[-1]+1]
            if len(mid) == 0:
                return
            else:
                mid = mid[0]
                idx_cyc = [_ for _, item in enumerate(self.selected) if item.mid == mid][0]
                self.curr_idx_cyc = idx_cyc

            # refresh the textbox and let the user label the cycles
            self.lb_naming_selected.set_val(f'Cycle {indices[0]}-{indices[-1]} selected!')
            self.txtbox.set_val('')
            # self.txtbox.on_submit(lambda text: self._submit_cycle_label(text, idx_cyc))

            # # display the textbox for the selected cycle(s)
            # # some mundane computation to know where to show the textbox
            # if self.ca_handler.display_mode == 0:
            #     _x = indices[0] + (indices[-1] - indices[0]) // 2
            # else:
            #     _x = self.data.iloc[indices[0], 'time'] + \
            #          (self.data.iloc[indices[-1], 'time'] - self.data.iloc[indices[0], 'time']) // 2
            # tb_x, tb_y = self.fig.transFigure.inverted().transform(ax.transData.transform((_x, event.ydata)))
            #
            # # if there's no txtbox already displayed, create one to receive labels
            # has_txtbox = any([True for _txt in self.txtbox_names if _txt.ax.get_position().xmin == tb_x])
            # if not has_txtbox:
            #     txtbox_ax = plt.axes([tb_x, tb_y, 0.08, 0.03])
            #     txtbox = TextBox(ax=txtbox_ax, label='Label')
            #     txtbox.on_submit(lambda text: self._submit_cycle_label(text, idx_cyc))
            #     self.txtbox_names.append(txtbox)

    def _submit_cycle_label(self, text):
        """

        Parameters
        ----------
        text

        Returns
        -------

        """
        if self.curr_idx_cyc is not None:
            self.selected[self.curr_idx_cyc].mlabel = text

    def _cycle_unselection(self, event, ax):
        """
        Unselect the cycles using right click

        Parameters
        ----------
        event
        ax

        Returns
        -------

        """
        # index of the selected axes
        idx_ax = self.fig.axes.index(ax)

        # if the selected axis is the one that contains real data
        if idx_ax in self.ax_ext_map:
            ext = self.ax_ext_map[idx_ax]

            # WHOLE selection mode
            if self.select_mode == 0:
                mids = [mk.mid for mk in self.current_query.candidates[ext].markers]
                self._clear_whole(ax=ax, mids=mids, ext=ext)
                ax.set_ylabel(ext)
            # INDIVIDUAL selection mode
            else:
                # indices of selected cycle based on xdata
                idx, _ = self._get_cycle_indices_on_click(x=event.xdata, y=self.y[ext])
                if idx is None:
                    return
                # xmin and xmax bound to display the selection span
                xmin, xmax = (idx[0], idx[-1]) if self.ca_handler.display_mode == 0 \
                    else (self.data.iloc[idx[0], 'time'], self.data.iloc[idx[-1], 'time'])
                #  retrieve the selected marker
                ext_markers = self.current_query.candidates[ext].markers
                mid = [mk.mid for mk in ext_markers if mk.mstart == idx[0] and mk.mend == idx[-1]]
                if len(mid) != 0:
                    self._clear_one_single_cycle(ax, mid[0], xmin, xmax)

    def _clear_whole(self, ax, mids, ext):
        """
        Clean all cycles of one extractor

        Returns
        -------

        """
        # remove all selected items of this extractor from the list
        self.selected = [item for item in self.selected if item.mid not in mids]

        # refresh the interface
        ax.clear()
        if self.ca_handler.display_mode == 0:
            _plot_by_point(ax=ax, data=self.data, atts=self.atts, y=self.y[ext])
        else:
            _plot_by_time(ax=ax, data=self.data, atts=self.atts, y=self.y[ext])

    def _clear_one_single_cycle(self, ax, mid, xmin, xmax):
        """
        Clear one cycle

        Returns
        -------

        """
        # remove this cycle from the list
        self.selected = [item for item in self.selected if item.mid != mid]

        # remove the highlight rectangle from the axis
        to_remove = [r for r in ax.get_children() if type(r) is Polygon
                     and r.get_xy()[0][0] == xmin and r.get_xy()[2][0] == r.get_xy()[2][0] == xmax]
        for tr in to_remove:
            tr.remove()

    def _cycle_selection(self, event, ax):
        """

        Parameters
        ----------
        event
        ax

        Returns
        -------

        """
        # index of the selected axes
        idx_ax = self.fig.axes.index(ax)

        # if the selected axis contains data
        if idx_ax in self.ax_ext_map:
            ext = self.ax_ext_map[idx_ax]               # index of the selected extractor

            # WHOLE mode
            if self.select_mode == 0:
                # TODO disallow selecting cycles from multiple extractors if those from one has already been chosen
                # restraint the user to select cycles from ONE extractor only
                indexes = get_cycles(self.y[ext])
                for idx in indexes:
                    # depending on whether we're in point-based or time-based mode
                    xmin, xmax = (idx[0], idx[-1]) if self.ca_handler.display_mode == 0 \
                        else (self.data.loc[idx[0], 'time'], self.data.iloc[idx[-1], 'time'])
                    ax.axvspan(xmin, xmax, edgecolor='k', linestyle='--', linewidth=1.5, facecolor=highlight)
                # retrieve the selected markers
                self.selected = [SelectedMarker(mid=mk.mid, mlabel='')
                                 for mk in self.current_query.candidates[ext].markers]
            # INDIVIDUAL mode
            else:
                # TODO disallow selecting cycles from multiple extractors if those from one has already been chosen
                # index of selected cycle based on xdata
                idx, _ = self._get_cycle_indices_on_click(x=event.xdata, y=self.y[ext])
                if idx is None:
                    return
                # xmin and xmax bound to display the selection span
                xmin, xmax = (idx[0], idx[-1]) if self.ca_handler.display_mode == 0 \
                    else (self.data.iloc[idx[0], 'time'], self.data.iloc[idx[-1], 'time'])
                ax.axvspan(xmin, xmax, edgecolor='k', linestyle='--', linewidth=1.5, facecolor=highlight)
                #  retrieve the selected marker
                ext_markers = self.current_query.candidates[ext].markers
                mid = [mk.mid for mk in ext_markers if mk.mstart >= idx[0] and mk.mend <= idx[-1]+1]
                if len(mid) != 0:
                    self.selected.append(SelectedMarker(mid=mid[0], mlabel=''))

    def _get_cycle_indices_on_click(self, x, y):
        """
        Retrieves the indices of selected cycles (in INDIVIDUAL selection mode)
        """
        idx_cycles = get_cycles(y)
        if self.ca_handler.display_mode == 0:  # if we're displaying data by point
            for i, idx in enumerate(idx_cycles):
                if idx[0] <= x <= idx[-1]:
                    return idx, i
            return None, None
        else:  # else, if we're displaying data by time
            for i, idx in enumerate(idx_cycles):
                if self.data.iloc[idx[0], 'time'] <= x <= self.data.iloc[idx[-1], 'time']:
                    return idx, i
            return None, None
    #endregion

    #region Selecting the segmentation mode (radio button)
    def _change_select_mode(self, label):
        """Change the selection mode as the user switches radio button options"""
        if label == 'Entire segmentation':
            self.select_mode = 0
            self.ca_handler.select_mode = 0
        else:
            self.select_mode = 1
            self.ca_handler.select_mode = 1
    #endregion

    #region Name the cycles
    def _turn_on_off_naming(self, label):
        """
        Turn on or off the naming mode (check button)
        """
        idx = self.naming_opt_labels.index(label)
        if label == 'Label cycles':
            check = self.cbx_naming.lines[idx]
            if check[0].get_visible() and check[1].get_visible():
                self.is_naming = True
                self.txtbox.active = True
                self.lb_naming_selected.set_val('Cycle labeling enabled!')
            else:
                self.is_naming = False
                self.txtbox.active = False
                self.lb_naming_selected.set_val('Cycle labeling disabled')
        else:
            self.is_naming = False
            self.lb_naming_selected.set_val('Cycle labeling disabled')
    #endregion
