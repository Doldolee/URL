from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import torch
from tqdm import tqdm
from collections import deque
from tqdm import tqdm
from typing import List, Union

## only extracts background
def extract_background_next_notes(next_notes, notes, done, missing='no clinical note'):
    """
    next_notes: list of str, aligned with `done`
    notes:      list of str, same length (unused, for signature consistency)
    done:       list of bool/int, True at episode end
    missing:    placeholder for missing notes

    returns: list of str, each entry is either "[context] first_valid_next_note"
             or the missing placeholder if no valid next_note seen in the current episode
    """
    backgrounds = []
    first_valid = None

    for nxt, cur, is_done in zip(next_notes, notes, done):
        # Use the first valid next_note in the episode as its background.
        if first_valid is None and nxt != missing:
            first_valid = nxt

        # Extract the background, returning missing if it is unavailable.
        if first_valid is None:
            backgrounds.append(missing)
        else:
            backgrounds.append(f"[context] {first_valid}")

        # Reset the background at the end of the episode.
        if is_done:
            first_valid = None

    return backgrounds
    
# only extracts background
def extract_background_states(notes, done, missing='no clinical note'):
    """
    notes: list of str
    done: list of bool/int, True at episode boundaries
    missing: placeholder for missing notes
    returns: list of str, each entry is either "[context] first_valid_note" or the missing placeholder
    """
    backgrounds = []
    first_valid = None

    for note, is_done in zip(notes, done):
        # Use the first valid note as the background.
        if first_valid is None and note != missing:
            first_valid = note

        # Add the background, using the missing placeholder when it is unavailable.
        if first_valid is None:
            backgrounds.append(missing)
        else:
            backgrounds.append(f"[context] {first_valid}")

        # Reset the background at the end of the episode.
        if is_done:
            first_valid = None

    return backgrounds

## background + window 3 stack
def bg_stack_next_note(
    next_notes: List[str],
    done: List[Union[bool, int]],
    sep: str = ' | ',
    missing: str = 'no clinical note'
) -> List[str]:
    """
    Episode-aware stacking for next_notes, including only t-1 and t-2 within the same episode.
    """
    stacked = []
    first_valid = None
    last_reset = -1

    for i, (nxt, flag) in enumerate(zip(next_notes, done)):
        flag = bool(flag)
        is_first = False

        # Detect first valid note in episode
        if first_valid is None and nxt != missing:
            first_valid = nxt
            is_first = True

        # Collect t-2, t-1 valid notes within current episode
        suffix = []
        for idx in (i-2, i-1):
            if idx > last_reset and 0 <= idx < len(next_notes):
                prev = next_notes[idx]
                if prev != missing and prev != first_valid and prev not in suffix:
                    suffix.append(prev)

        # Build output
        if first_valid is None:
            out = nxt
        else:
            if is_first:
                out = f"[current status] {nxt}"
            else:
                prefix = f"[background] {first_valid}"
                if suffix:
                    prefix += sep + sep.join(suffix)
                if nxt != missing:
                    out = f"{prefix} || [current status] {nxt}"
                else:
                    out = prefix

        stacked.append(out)

        # Reset at episode boundary
        if flag:
            first_valid = None
            last_reset = i

    return stacked

## background + window 3 stack
def bg_stack_note(notes: List[str],
                          done: List[Union[bool, int]],
                          sep: str = ' | ',
                          missing: str = 'no clinical note'
                         ) -> List[str]:
    stacked = []
    first_valid = None

    for i, (note, flag) in enumerate(zip(notes, done)):
        flag = bool(flag)
        
        # At the start of an episode or before its background has been set.
        if first_valid is None:
            if note != missing:
                first_valid = note
                out = f"[current status] {note}"
            else:
                out = note
            stacked.append(out)
            if flag:
                first_valid = None
            continue
        
        # Extract only valid notes at indices t-2 and t-1 for the suffix.
        prev_idxs = [i-2, i-1]
        suffix = []
        for idx in prev_idxs:
            if 0 <= idx < len(notes):
                prev_note = notes[idx]
                if prev_note != missing and prev_note != first_valid:
                    if prev_note not in suffix:
                        suffix.append(prev_note)
        
        prefix = f"[background] {first_valid}"
        
        if note != missing:
            # Use the current note as the current status.
            prefix_ext = prefix + (sep + sep.join(suffix) if suffix else "")
            out = f"{prefix_ext} || [current status] {note}"
        else:
            # Handle a missing note at this time step.
            out = prefix + (f" || {sep.join(suffix)}" if suffix else "")
        
        stacked.append(out)
        if flag:
            first_valid = None
    
    return stacked
    
## background
def impute_next_notes_with_background(next_notes, notes, done, missing='no clinical note'):
    """next_notes: List of strings aligned with done.
    notes: Unused list retained for a consistent signature.
    done: Boolean or integer flags marking episode ends.
    missing: Placeholder for missing notes.
    
    For each next_note, return only '[background] next_note' for the episode's first valid note. For later valid notes, return '[background] first_valid || next_note'. For missing notes, return only '[background] first_valid'. Before a valid background exists, return the original next_note."""
    imputed = []
    first_valid = None

    for nxt, cur, is_done in zip(next_notes, notes, done):
        # Check whether this is the episode's first valid next_note.
        is_first = False
        if first_valid is None and nxt != missing:
            first_valid = f"[background] {nxt}"
            is_first = True

        if first_valid is not None:
            if is_first:
                # For the first valid note, return only the background.
                imputed_note = first_valid
            else:
                if nxt == missing:
                    # For missing notes, return only the background.
                    imputed_note = first_valid
                else:
                    # For subsequent valid notes, combine the background with the original note.
                    imputed_note = f"{first_valid} || {nxt}"
        else:
            # Before the first valid note, keep the original note.
            imputed_note = nxt

        imputed.append(imputed_note)

        # Reset the background at episode boundaries.
        if is_done:
            first_valid = None

    return imputed

## background
def impute_notes_with_background(notes, done, missing='no clinical note'):
    """Add episode backgrounds to a list of note strings. done marks episode boundaries and missing identifies absent notes. Missing notes contribute only their background; their placeholder text is omitted."""
    imputed = []
    first_valid = None

    for note, is_done in zip(notes, done):
        # Store the first valid note in an episode with a '[background]' prefix.
        if first_valid is None and note != missing:
            first_valid = f"[background] {note}"

        if first_valid is not None:
            if note == missing:
                # For missing notes, retain only the background.
                imputed_note = first_valid
            else:
                # Combine each valid note with the background.
                imputed_note = f"{first_valid} || {note}"
        else:
            # Keep the original note until a valid background is available.
            imputed_note = note

        imputed.append(imputed_note)

        # Reset the background for the next step after an episode ends.
        if is_done:
            first_valid = None

    return imputed
    
## simple impute
def impute_notes(notes, done, missing='no clinical note'):
    """Impute a list of note strings. done contains episode-boundary flags and missing identifies absent notes. Return the imputed list."""
    imputed = []
    last_valid = None

    for note, is_done in zip(notes, done):
        if note != missing:
            # Record a valid note and retain it as the previous note.
            imputed.append(note)
            last_valid = note
        else:
            # Replace a missing note with the most recent valid note, if available.
            imputed.append(last_valid if last_valid is not None else note)

        # Reset last_valid for the next step after an episode ends.
        if is_done:
            last_valid = None

    return imputed

## simple impute
def impute_next_notes(next_notes, notes, done, missing='no clinical note'):
    """
    next_notes: list of str, aligned with `done`, where next_notes[i] is the note at the next time step
    notes:      list of str, aligned one step behind next_notes (so notes[i] is the “current” note for next_notes[i])
    done:       list of bool/int, True (or 1) at the end of each episode, aligned with next_notes
    missing:    the placeholder string indicating a missing clinical note

    Returns a new list where each missing next_note is imputed by:
      1) the current note at the same index, if available
      2) else the most recent valid imputed note within the same episode
      3) else left as missing (if no valid note seen yet)
    """
    imputed = []
    last_valid = None

    for idx, (nxt, cur, is_done) in enumerate(zip(next_notes, notes, done)):
        if nxt != missing:
            imputed.append(nxt)
            last_valid = nxt
        else:
            # 1) First try to fill from the current note at the same index.
            if cur != missing:
                imputed.append(cur)
                last_valid = cur
            # 2) Otherwise use the most recent valid note in the same episode.
            else:
                imputed.append(last_valid if last_valid is not None else missing)

        # Reset last_valid for the next index after an episode ends.
        if is_done:
            last_valid = None

    return imputed

def stack_notes(notes, done, window=3, sep=' | ', missing='no clinical note'):
    """Stack notes over a time window within each episode.
    
    Args:
        notes: List of note strings.
        done: Boolean or integer episode-boundary flags.
        window: Number of time steps to stack; default 3.
        sep: Separator between notes; default ' | '.
        missing: Missing-note placeholder; default 'no clinical note'.
    
    Return a list of stacked strings excluding missing notes."""
    stacked = []
    dq = deque(maxlen=window)

    for note, is_done in tqdm(zip(notes, done)):
        # 1) Add the current note, including missing notes to preserve the time window.
        dq.append(note)

        # 2) Combine only the nonmissing notes in the deque.
        valid = [n for n in dq if n != missing]
        stacked.append(sep.join(valid) if valid else '')

        # 3) Reset the window at episode boundaries.
        if is_done:
            dq.clear()

    return stacked

    
def stack_next_notes(next_notes, notes, done, window=3, sep=' | ', missing='no clinical note'):
    """Stack next_notes over recent time steps while respecting episode boundaries.
    
    Args:
        next_notes: Next-time-step notes, with next_notes[i] == notes[i+1].
        notes: Current notes aligned with done.
        done: Boolean or integer flags marking episode ends, aligned with notes.
        window: Number of time steps to stack; default 3.
        sep: Separator between notes; default ' | '.
        missing: Missing-note placeholder; default 'no clinical note'.
    
    At each index, join the valid next_notes from the most recent window using sep."""
    stacked = []
    dq = deque(maxlen=window)

    for nxt, cur, is_done in tqdm(zip(next_notes, notes, done)):
        # 1) Clear the buffer immediately at an episode boundary defined by the current note.
        if is_done:
            dq.clear()

        # 2) Add the next-time-step note to the buffer.
        dq.append(nxt)

        # 3) Combine only valid, nonmissing notes.
        valid = [n for n in dq if n != missing]
        stacked.append(sep.join(valid) if valid else '')

    return stacked

def custom_collate_fn(batch):
    # Each batch item is a tuple of (note, dem, state, action, length, time, reward).
    notes, dems, states, actions, lengths, times, rewards = zip(*batch)
    # Keep strings in lists and combine the numeric data into tensors.
    batch_dem = torch.stack(dems)
    batch_states = torch.stack(states)
    batch_actions = torch.stack(actions)
    batch_lengths = torch.stack(lengths)
    batch_times = torch.stack(times)
    batch_rewards = torch.stack(rewards)
    return np.array(notes, dtype=object), batch_dem, batch_states, batch_actions, batch_lengths, batch_times, batch_rewards

class CustomDataset(Dataset):
    def __init__(self, train_note, train_dem, train_states, train_actions, train_lengths, train_times, train_rewards):
        """Args:
            train_note: List or array of strings.
            train_dem: Numeric demographic data.
            train_states: Numeric state data.
            train_actions: Numeric action data.
            train_lengths: Numeric sequence lengths.
            train_times: Numeric timestamp data.
            train_rewards: Numeric reward data."""
        self.train_note = train_note  # Keep the string data unchanged.
        
        # Convert the remaining data to torch tensors, adjusting the dtype if needed.
        self.train_dem = train_dem.clone().detach() if isinstance(train_dem, torch.Tensor) else torch.tensor(train_dem, dtype=torch.float32)
        self.train_states = train_states.clone().detach() if isinstance(train_states, torch.Tensor) else torch.tensor(train_states, dtype=torch.float32)
        self.train_actions = train_actions.clone().detach() if isinstance(train_actions, torch.Tensor) else torch.tensor(train_actions, dtype=torch.float32)
        self.train_lengths = train_lengths.clone().detach() if isinstance(train_lengths, torch.Tensor) else torch.tensor(train_lengths, dtype=torch.float32)
        self.train_times = train_times.clone().detach() if isinstance(train_times, torch.Tensor) else torch.tensor(train_times, dtype=torch.float32)
        self.train_rewards = train_rewards.clone().detach() if isinstance(train_rewards, torch.Tensor) else torch.tensor(train_rewards, dtype=torch.float32)

    def __len__(self):
        return len(self.train_note)

    def __getitem__(self, index):
        # Return strings unchanged and index the remaining tensor fields.
        note = self.train_note[index]
        dem = self.train_dem[index]
        state = self.train_states[index]
        action = self.train_actions[index]
        length = self.train_lengths[index]
        time = self.train_times[index]
        reward = self.train_rewards[index]
        return note, dem, state, action, length, time, reward
        
class ReplayBuffer(object):
    def __init__(self, 
                 state_dim,
                 batch_size,
                 buffer_size,
                 device,
                 data_path,
                 buffer_path
                 ):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.data_path = data_path
        self.buffer_path = buffer_path
        
    
        self.ptr = 0
        self.crt_size = 0

        self.note = np.empty((self.max_size, 1), dtype=object)
        self.next_note = np.empty((self.max_size, 1), dtype=object)

        self.state = np.zeros((self.max_size, state_dim))
        self.next_state = np.array(self.state)
        self.action = np.zeros((self.max_size, 1))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, notes, next_notes, states, action, next_state, reward, done):
        self.note[self.ptr] = notes
        self.next_note[self.ptr] = next_notes
        self.state[self.ptr] = states
        self.next_state[self.ptr] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)
        
    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)

        return(
            self.note[ind],
            self.next_note[ind],
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
     
    def save(self, only_test_set = False):
        if only_test_set:
            flag = "test"
        else:
            flag = "train_val"
        np.save(f"{self.buffer_path}{flag}_note.npy", self.note[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_next_note.npy", self.next_note[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_state.npy", self.state[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_action.npy", self.action[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_done.npy", self.done[:self.crt_size])


    def load(self, size=-1, only_test_set = False):
        if only_test_set:
            flag = "test"
        else:
            flag = "train_val"

        reward_buffer = np.load(f"{self.buffer_path}{flag}_reward.npy")
      
        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.note[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_note.npy", allow_pickle=True)[:self.crt_size]
        self.next_note[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_next_note.npy", allow_pickle=True)[:self.crt_size]
        self.state[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.done[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_done.npy")[:self.crt_size]
        print(f"Replay Buffer loaded with {self.crt_size} elements.") # The allocated buffer size remains unchanged; only entries up to crt_size are filled.

        
    
    def load_initial_data(self, only_test_set = False):
    
        train_note, train_dem, train_states, train_actions, train_lengths, train_times, train_rewards = torch.load(self.data_path["train"], weights_only = False)
        train_dataset = CustomDataset(train_note, train_dem, train_states, train_actions, train_lengths, train_times, train_rewards)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        val_note, val_dem, val_states, val_actions, val_lengths, val_times, val_rewards = torch.load(self.data_path["val"], weights_only = False)
        val_dataset = CustomDataset(val_note, val_dem, val_states, val_actions, val_lengths, val_times, val_rewards)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        test_note, test_dem, test_states, test_actions, test_lengths, test_times, test_rewards = torch.load(self.data_path["test"], weights_only = False)
        test_dataset = CustomDataset(test_note, test_dem, test_states, test_actions, test_lengths, test_times, test_rewards)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        if only_test_set:
            all_loaders_list = [test_loader]
        else:
            all_loaders_list = [train_loader, val_loader]

        for idx, loader in enumerate(all_loaders_list):
            for note, dem, state, action, length, time, reward in tqdm(loader):
                # print(note.shape, dem.shape, state.shape, action.shape, length.shape, time.shape, reward.shape)
                note = note
                dem = dem.to(self.device)
                state = state.to(self.device)
                action = action.to(self.device)
                length = length.to(self.device)
                time = time.to(self.device)
                reward = reward.to(self.device)
            
                max_length = int(length.max().item())
                note = note[:,:max_length,:]
                state = state[:,:max_length,:]
                dem = dem[:,:max_length,:]
                action = action[:,:max_length,:]
                reward = reward[:,:max_length]

                cur_notes, next_notes = note[:,:-1,:], note[:,1:,:]   
                cur_states, next_states = state[:,:-1,:], state[:,1:,:]                
                cur_dem, next_dem = dem[:,:-1,:], dem[:,1:,:]
                cur_actions = action[:,:-1,:]
                cur_rewards = reward[:,1:] # This ordering is compatible with terminal-only rewards; revisit it before using intermediate rewards.
      
                for batch in range(cur_states.shape[0]):
                    for i_trans in range(cur_states.shape[1]):
                        done = cur_rewards[batch,i_trans] != 0 # A nonzero value marks the end of an episode and becomes True (1).
                        self.add(notes = cur_notes[batch,i_trans],
                                next_notes = next_notes[batch,i_trans],
                                states=torch.cat((cur_states[batch,i_trans],cur_dem[batch,i_trans]),dim=-1).cpu().numpy(), 
                                action=cur_actions[batch,i_trans].cpu().argmax().item(), # scalar
                                next_state=torch.cat((next_states[batch,i_trans], next_dem[batch,i_trans]), dim=-1).cpu().numpy(), 
                                reward=cur_rewards[batch,i_trans].cpu().item(), 
                                done=int(done.item()) # True represents termination (1); False is 0.
                                )
                        if done:
                            break
        print("only test set? : ", only_test_set)
        print(self.ptr, self.crt_size)
        self.save(only_test_set = only_test_set)