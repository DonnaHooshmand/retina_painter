"""
Copyright (C) 2023 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from dataclasses import dataclass
import os

@dataclass
class Event:
    """Class for keeping track of an interaction event."""
    name: str # event type i.e mouse_press
    time: float
    fname: int = 0


def is_pause(prev_event, period_len_s):
    # if the previous event is not mouse_press 
    # (maybe the user us just taking their time to draw something without lifting the mouse)
    if prev_event.name != 'mouse_press':
        # and the duration is over 20 seconds
        if period_len_s > 20:
            # then don't include the interaction - we consider this a pause in annotation
            return True

    # 'save_annotation' is called when navigating away from the file.
    # meaning that time since this was called was spent on other files
    if prev_event.name == 'save_annotation':
        return True
    return False

def get_annot_duration_s(events, fname):
    fname = os.path.splitext(fname)[0] # event object doesn't include file extension
    # we consider interaction as mouseup and mouse down events
    fname_events = [e for e in events if e.fname == fname and e.name in [
                    'mouse_press', 'mouse_release',
                    'save_annotation', 'update_file_end']]

    if len(fname_events) < 2: # must have both mouse_press and mouse_release
        return 0, 0
    click_count = 0
    # fname_events are already filtered by filename
    total_duration = 0
    most_recent_event = fname_events[0]
    for e in fname_events[1:]:
        period_between_events = e.time - most_recent_event.time
        
        # if the current event is mouse_up then we count this as a click.
        if e.name == 'mouse_release':
            click_count += 1
        if not is_pause(most_recent_event, period_between_events):
            total_duration += period_between_events
        most_recent_event = e
    return total_duration, click_count


def events_from_client_log(client_log_fpath):
    """Events of the client log that belong to a file.

    Lines that name no file (for example ``automatic_start_training``), and
    lines that are not events at all, are skipped. Before this they raised an
    error, which stopped the metrics plot and export in any project whose
    training had started automatically.
    """
    with open(client_log_fpath) as log_file:
        lines = log_file.readlines()
    events = []
    for l in lines:
        parts = l.strip().split(',')
        if len(parts) < 3:
            continue
        fname_parts = [p for p in parts if p.startswith('fname:')]
        if not fname_parts:
            continue
        try:
            event_time = float(parts[1])
        except ValueError:
            continue
        event_name = parts[2]
        fname = fname_parts[0].replace('fname:', '', 1)
        # because load image has .jpg name (perhaps) but save annotation has png
        # so let's just work with the file name without the extension.
        fname = os.path.splitext(fname)[0]
        events.append(Event(name=event_name, fname=fname, time=event_time))

    return events


def screen_time_per_fname_s(events, pause_s=120.0):
    """Time each file was on screen: ``{fname: (seconds, visits, paused_s)}``.

    A file is on screen from its ``update_file_end`` event until the next
    ``update_file_end`` of any file (the last one, until the last logged
    event). Unlike ``get_annot_duration_s`` this includes time spent looking
    without painting, which is most of the time on a scan with nothing to
    correct. A gap of more than ``pause_s`` seconds between two consecutive
    logged events is taken as time away from the screen and is reported
    separately, not counted.
    """
    totals = {}
    current = None
    previous_time = None
    for event in sorted(events, key=lambda e: e.time):
        if current is not None:
            gap = event.time - previous_time
            seconds, visits, paused = totals[current]
            if gap > pause_s:
                paused += gap
            else:
                seconds += gap
            totals[current] = (seconds, visits, paused)
        if event.name == 'update_file_end':
            current = event.fname
            seconds, visits, paused = totals.get(current, (0.0, 0, 0.0))
            totals[current] = (seconds, visits + 1, paused)
        previous_time = event.time
    return totals


def interaction_time_per_fname_s(client_log_fpath):
    """ estimate interaction time for each file based
        on the timing of mouse_press and mouse_release events
        logged whilst the user was interacting with each file 

        Example usage:
        interaction_times = interaction_time_per_fname_s(client_log_fpath='logs/client.csv')
     """
    events = events_from_client_log(client_log_fpath)     
    unique_fnames = list(set(e.fname for e in events))
    fname_times = {}
    for fname in unique_fnames:
        fname_times[fname] = get_annot_duration_s(events, fname)
    return fname_times
