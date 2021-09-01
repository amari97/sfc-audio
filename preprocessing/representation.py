import numpy as np
from librosa.feature import mfcc
from gmpy2 import digits
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

HOP_LENGTH = 0.01  # seconds
WINDOW_LENGTH = 0.025  # seconds
F_MIN = 64


def compute_mfcc(dataset: np.ndarray, high_res: bool = False, sr: int = 16000, duration: float = 1.0) -> np.ndarray:
    """Compute MFCC
    Args:
        dataset (np.ndarray): a numpy array containing the audio seuqences
        sr (int): the sampling rate (default 16000)
        high_res (bool): a boolean value to indicate how many MFCC coefficient we need (20 or 40)
        duration (float,optional): the duration of the sample (default 1.0). Padded with zeros if shorter
    """
    if dataset.ndim == 1:
        dataset = dataset.reshape((1, -1))
    nb_samples = dataset.shape[0]

    if high_res:
        bands = 40
    else:
        bands = 20

    spectrograms = np.zeros(
        (nb_samples, int(1+np.floor((duration-WINDOW_LENGTH)/HOP_LENGTH)), bands))
    for i in range(nb_samples):
        spectrograms[i, :, :] = mfcc(dataset[i, :], sr=sr, n_mfcc=bands, center=False,
                                     hop_length=int(HOP_LENGTH*sr), n_fft=int(WINDOW_LENGTH*sr), fmin=F_MIN, fmax=sr/2.0, window="hann").T

    return spectrograms


class Curve:
    """"Abstract basic Curve
    build_curve must be overidden
    reset must be overidden
    """

    def __init__(self, name: str, length: int, base: int) -> None:
        """
        Args:
            name (str): the name of the curve
            length (int): the minimum length of the curve
            base (int): the base of the curve
        """
        self.name = name
        self.base = base
        self.level = self.find_level(length, base)
        self.build_curve(self.level)

    def build_curve(self, level: int) -> None:
        self.X = []
        self.Y = []
        raise NotImplementedError

    def change_level(self, new_level: int) -> None:
        self.level = new_level
        self.reset()
        self.build_curve(new_level)

    def reset(self) -> None:
        raise NotImplementedError

    def find_level(self, length: int, base: int) -> int:
        return int(np.ceil(np.log(length)/np.log(self.base**2)))

    def plot(self, save=False, figsize=(8, 6), ax=None, path="figure") -> None:
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        ax.plot(self.X, self.Y)
        ax.set_title("{} curve of order {}".format(self.name, self.level))
        if save:
            plt.savefig(
                os.path.join(path, "{}_curve_order_{}.png".format(self.name, self.level)), dpi=150)

    def __str__(self) -> str:
        return "{} curve of order {}".format(self.name, self.level)


def choose_curve(name_curve: str, length: int, method: str = "sfc") -> Curve:
    """Construct the curve given its name and the minimal length of the curve (length<4**level). 
    The base of the curve can be changed for the Z curve"""
    if method != "sfc":
        return None
    if name_curve == "Hilbert":
        return Hilbert_Curve(length=length)
    if name_curve == "Z":
        return Z_Curve(length=length)
    if name_curve == "Gray":
        return Gray_Curve(length=length)
    if name_curve == "OptR":
        return OptR_Curve(length=length)
    if name_curve == "H":
        return H_Curve(length=length)
    if name_curve == "Sweep":
        return Sweep_Curve(length=length)
    if name_curve == "Scan":
        return Scan_Curve(length=length)
    if name_curve == "Diagonal":
        return Diagonal_Curve(length=length)
    raise ValueError("Unknown curve")


def find_dimensions(curve: Curve, method: str, sr: int, length: int) -> Tuple[int, int]:
    """
    Compute the dimensions of the input
    Args:
        curve (str): the curve object (can be none if method is mfcc)
        length (int): the length (fixed) of the audio sequence
        method (str): the method used. Either mfcc or sfc
        sr (int): the sampling rate
    Remark:
        We use HOP_LENGTH = 0.01 seconds and WINDOW_LENGTH = 0.025 seconds to compute the input dimensions in the case of mfcc
    """
    assert method in ["mfcc", "sfc"]

    if method == "sfc":
        x_size, y_size = curve.base**curve.level, curve.base**curve.level
    else:
        x_size, y_size = int(
            1+np.floor((length/sr-WINDOW_LENGTH)/HOP_LENGTH)), 40
    return x_size, y_size


class Gray_Curve(Curve):
    """
    Compute the mapping of the Gray curve [1]
    Reference:
        [1] C. Faloutsos,   “Multiattribute Hashing Using GrayCodes,” in Proc. SIGMOD, 1986, pp. 227–238
    """

    def __init__(self, length: int = 4**4) -> None:
        """
        Args:
            length (int): the minimum length of the curve (length<4**level)
        """
        self.L_system = {1: [1, 4, 1], 2: [2, 5, 2],
                         -1: [-1, -4, -1], -2: [-2, -5, -2],
                         3: [3, 2, -3, 1, -3, -2, 3],
                         -3: [-3, -2, 3, -1, 3, 2, -3],
                         4: [4], -4: [-4], 5: [5], -5: [-5]}
        super().__init__("Gray", length, 2)

    def build_curve(self, level: int) -> None:
        seq = self._compute_seq([3], level)
        # remove -3,3, (only -1,1,-2,2,4,-4,5,-5 for encoding the sequence)
        # 1=4,2=5
        # -1 = South, 1=North, -2=West, 2=East
        filtered = list(filter(lambda x: x not in [-3, 3], seq))
        X, Y = [], []
        x, y = 0, 0
        prev_dx, prev_dy = 0, 0
        xold, yold = 0, 0
        for action in filtered:
            dx, dy = 0, 0
            if action == 1 or action == 4:
                dx = 1
            elif action == -1 or action == -4:
                dx = -1
            elif action == 2 or action == 5:
                dy = 1
            elif action == -2 or action == -5:
                dy = -1
            x += dx
            y += dy

            # add only if next direction is different (allow jumps)
            if prev_dx != dx or prev_dy != dy:
                X.append(xold)
                Y.append(yold)

            # store old values
            prev_dx, prev_dy = dx, dy
            xold, yold = x, y

        # add the last point in the bottom right corner
        X.append(X[-1])
        Y.append(Y[-1]-1)

        self.X = X
        self.Y = Y

    def _compute_seq(self, seq: List, level: int) -> List:
        """Expands the seq recursively using the L-system (inplace replacement using the grammar)"""
        if level == 0:
            return seq
        else:
            new_seq = []
            # extend the new sequence using the rules of the L-system
            for l in seq:
                new_seq.extend(self.L_system[l])
            return self._compute_seq(new_seq, level-1)

    def reset(self) -> None:
        self.X = []
        self.Y = []


class OptR_Curve(Curve):
    def __init__(self, length: int = 4**4) -> None:
        """Build a recurcive curve that is optimal (wrt. the number of seek operation)
        Rules are build based on [1].
        Args:
            length (int): the minimum length of the curve (length<4**level)

        References:

            [1]     Tetsuo Asano, Desh Ranjan, Thomas Roos, Emo Welzl, Peter Widmayer,
                    Space-filling curves and their use in the design of geometric data structures,
                    Theoretical Computer Science,Volume 181, Issue 1,1997,Pages 3-15,
                    ISSN 0304-3975,https://doi.org/10.1016/S0304-3975(96)00259-9

        """
        # 1 move along the x axis (+=right, -=left )
        # 2 move along the y axis (+=up, -=down ),
        # 3 and 4 for the diagonals (3=up right,-3=down left, 4=down right, -4=up left)
        # N/S/E/W is for the orientation of the tile
        # R stands for Reverse: if we need to reverse to direction of the tile (start from the exit)
        partial = {1: [1], -1: [-1],
                   2: [2], -2: [-2],
                   3: [3], -3: [-3],
                   4: [4], -4: [-4],
                   "A1N": ["D2N", 1, "B1ER", -4, "C1W", 1, "B2SR"],
                   "A1W": ["D2W", 2, "B1NR", -3, "C1S", 2, "B2ER"],
                   "A1E": ["D2E", -2, "B1SR", 3, "C1N", -2, "B2WR"],
                   "A1S": ["D2S", -1, "B1WR", 4, "C1E", -1, "B2NR"],
                   "A2N": ["B1ER", 2, "C2N", 4, "B2SR", 2, "D1W"],
                   "A2W": ["B1NR", -1, "C2W", 3, "B2ER", -1, "D1S"],
                   "A2E": ["B1SR", 1, "C2E", -3, "B2WR", 1, "D1N"],
                   "A2S": ["B1WR", -2, "C2S", -4, "B2NR", -2, "D1E"],
                   "B1N": ["D1ER", 2, "C2N", 1, "B1N", -2, "B2WR"],
                   "B1W": ["D1NR", -1, "C2W", 2, "B1W", 1, "B2SR"],
                   "B1E": ["D1SR", 1, "C2E", -2, "B1E", -1, "B2NR"],
                   "B1S": ["D1WR", -2, "C2S", -1, "B1S", 2, "B2ER"],
                   "B2N": ["B1ER", 2, "B2N", 1, "C1N", -2, "D2WR"],
                   "B2W": ["B1NR", -1, "B2W", 2, "C1W", 1, "D2SR"],
                   "B2E": ["B1SR", 1, "B2E", -2, "C1E", -1, "D2NR"],
                   "B2S": ["B1WR", -2, "B2S", -1, "C1S", 2, "D2ER"],
                   "C1N": ["A2SR", 2, "B1W", 1, "A1E", -2, "B2WR"],
                   "C1W": ["A2ER", -1, "B1S", 2, "A1N", 1, "B2SR"],
                   "C1E": ["A2WR", 1, "B1N", -2, "A1S", -1, "B2NR"],
                   "C1S": ["A2NR", -2, "B1E", -1, "A1W", 2, "B2ER"],
                   "C2N": ["B1ER", 2, "A2N", 1, "B2E", -2, "A1WR"],
                   "C2W": ["B1NR", -1, "A2W", 2, "B2N", 1, "A1SR"],
                   "C2E": ["B1SR", 1, "A2E", -2, "B2S", -1, "A1NR"],
                   "C2S": ["B1WR", -2, "A2S", -1, "B2W", 2, "A1ER"],
                   "D1N": ["D1ER", 2, "A2N", 1, "C2E", -2, "A2E"],
                   "D1W": ["D1NR", -1, "A2W", 2, "C2N", 1, "A2N"],
                   "D1E": ["D1SR", 1, "A2E", -2, "C2S", -1, "A2S"],
                   "D1S": ["D1WR", -2, "A2S", -1, "C2W", 2, "A2W"],
                   "D2N": ["A1N", 2, "C1W", 1, "A1E", -2, "D2WR"],
                   "D2W": ["A1W", -1, "C1S", 2, "A1N", 1, "D2SR"],
                   "D2E": ["A1E", 1, "C1N", -2, "A1S", -1, "D2NR"],
                   "D2S": ["A1S", -2, "C1E", -1, "A1W", 2, "D2ER"]}
        self.L_system = partial.copy()
        # add reverse tiles
        for key, rule in partial.items():
            if isinstance(key, str):
                new_rule = []
                # iterate over all elements of the rule
                for i, v in enumerate(rule):
                    if isinstance(v, int):
                        # change the sign of the directions
                        new_rule.insert(0, -v)
                    else:
                        # if already in reverse mode, we remove R
                        if v.endswith("R"):
                            new_rule.insert(0, v[:3])
                        else:
                            # otherwise we add R
                            new_rule.insert(0, v+"R")
                # add the rule to the L-system
                self.L_system[key+"R"] = new_rule

        super().__init__("OptR", length, 2)

    def build_curve(self, level: int) -> None:
        seq = self._compute_seq(["A1N"], level)
        # -1 = South, 1=North, -2=West, 2=East
        # 3 and 4 for the diagonals (3=NE,-3=SW, 4=SE, -4=NW)
        filtered = list(
            filter(lambda x: x in [1, -1, 2, -2, -3, 3, 4, -4], seq))
        X, Y = [0], [0]
        x, y = 0, 0
        for action in filtered:
            dx, dy = 0, 0
            if action in [1, 3, 4]:
                dx = 1
            if action in [-1, -3, -4]:
                dx = -1
            if action in [2, 3, -4]:
                dy = 1
            if action in [-2, -3, 4]:
                dy = -1
            x += dx
            y += dy

            X.append(x)
            Y.append(y)
        self.X = X
        self.Y = Y

    def _compute_seq(self, seq: List, level: int) -> List:
        """Expands the seq recursively using the L-system (inplace replacement using the grammar)"""
        if level == 0:
            return seq
        else:
            new_seq = []
            # extend the new sequence using the rules of the L-system
            for l in seq:
                new_seq.extend(self.L_system[l])
            return self._compute_seq(new_seq, level-1)

    def reset(self) -> None:
        self.X = []
        self.Y = []


class Hilbert_Curve(Curve):
    """Builds the Hilbert curve [1].
    References:
        [1] Hilbert, D. (1935).Über die stetige Abbildung einer Linie auf ein Flächenstück, pages 1–2.Springer Berlin Heidelberg, Berlin, Heidelberg.
    """

    def __init__(self, length: int = 4**4) -> None:
        """
        Args:
            length (int): the minimum length of the curve (length<4**level)
        """
        self.L_system = {1: [1], 2: [2],
                         -1: [-1], -2: [-2],
                         3: [4, 2, 3, 1, 3, -2, -4],
                         -3: [-4, -2, -3, -1, -3, 2, 4],
                         4: [3, 1, 4, 2, 4, -1, -3],
                         -4: [-3, -1, -4, -2, -4, 1, 3]}
        super().__init__("Hilbert", length, 2)

    def build_curve(self, level: int) -> None:
        seq = self._compute_seq([3], level)
        # remove -3,3,-4,4 (only -1,1,-2,2 for encoding the sequence)
        # -1 = West, 1=East, -2=South, 2=Nord
        filtered = list(filter(lambda x: x not in [-3, 3, -4, 4], seq))
        X, Y = [0], [0]
        x, y = 0, 0
        for action in filtered:
            dx, dy = 0, 0
            if action == 1:
                dx = 1
            elif action == -1:
                dx = -1
            elif action == 2:
                dy = 1
            elif action == -2:
                dy = -1
            x += dx
            y += dy

            X.append(x)
            Y.append(y)
        self.X = X
        self.Y = Y

    def _compute_seq(self, seq: List, level: int) -> None:
        if level == 0:
            return seq
        else:
            new_seq = []
            # extend the new sequence using the rules of the L-system
            for l in seq:
                new_seq.extend(self.L_system[l])
            return self._compute_seq(new_seq, level-1)

    def reset(self) -> None:
        self.X = []
        self.Y = []


class Z_Curve(Curve):
    """Builds the Z curve [1]
    References:
        [1] Morton, G. M. (1966). A computer oriented geodetic data base and a new technique in filesequencing. Technical report, IBM Ltd., Ottawa
    """

    def __init__(self, length: int = 4**4) -> None:
        """
        Args:
            length (int): the minimum length of the curve (length<base**(2*level))
        """
        super().__init__("Z", length, 2)

    def build_curve(self, level: int) -> None:
        self._z_curve(level)

    def _z_index_to_2d(self, index: int) -> None:
        # binary representation (in reverse order)
        bin_index = digits(index, self.base)[::-1]
        # split between odd and even positions (and revert order)
        x, y = int("0"+bin_index[1::2][::-1],
                   self.base), int("0"+bin_index[0::2][::-1], self.base)
        return x, y

    def _z_curve(self, level: int) -> None:
        """Compute the Z-order curve using bit interleaving"""
        X, Y = [], []
        side_length = self.base**level
        # iterates over the whole sequence
        for i in range(side_length**2):
            x, y = self._z_index_to_2d(i)
            X.append(x)
            Y.append(y)

        self.X = X
        self.Y = Y

    def reset(self) -> None:
        self.X = []
        self.Y = []


class H_Curve(Curve):
    """Implements the H curve described in [1]

    References:

        [1] Rolf Niedermeier, Klaus Reinhardt, Peter Sanders,
            Towards optimal locality in mesh-indexings,
            Discrete Applied Mathematics,Volume 117, Issues 1–3,2002,Pages 211-237,
            ISSN 0166-218X,https://doi.org/10.1016/S0166-218X(00)00326-7.
    """

    def __init__(self, length: int = 4**4):
        """
        Args:
            length (int): the minimum length of the curve (length<4**level)
        """
        self.number_call = 0
        self.L_system = {1: [1], 2: [2],
                         -1: [-1], -2: [-2],
                         3: [3, 2, 4, 1, -6, 1, 3],
                         -3: [-3, -1, 6, -1, -4, 2, -3],
                         4: [4, -1, -5, -1, 3, 2, 4],
                         -4: [-4, -2, -3, 1, 5, 1, -4],
                         5: [5, 1, -4, 1, 6, 2, 5],
                         -5: [-5, -2, -6, -1, 4, -1, -5],
                         6: [6, 2, 5, -1, -3, -1, 6],
                         -6: [-6, 1, 3, 1, -5, -2, -6]}
        super().__init__("H", length, 2)

    def build_curve(self, level: int) -> None:
        seq = self._compute_seq([3], level-1)
        # 2=3=4=5=6 and -2=-3=-4=-5=-6
        # -2 = South, 2=North, -1=West, 1=East
        X, Y = [0], [0]
        x, y = 0, 0
        for action in seq:
            dx, dy = 0, 0
            if action == 1:
                dx = 1
            elif action == -1:
                dx = -1
            elif action in [2, 3, 4, 5, 6]:
                dy = 1
            elif action in [-2, -3, -4, -5, -6]:
                dy = -1
            x += dx
            y += dy

            # add the last point in the bottom right corner
            X.append(x)
            Y.append(y)

        # add bottom triangle
        X.extend(list(map(lambda x: -x+2**(level)-1, X)))
        Y.extend(list(map(lambda x: -x+2**(level)-1, Y)))

        self.X = X
        self.Y = Y


    def _compute_seq(self, seq: List, level: int) -> List:
        if level == 0:
            return seq
        else:
            new_seq = []
            # extend the new sequence using the rules of the L-system
            for l in seq:
                new_seq.extend(self.L_system[l])
            return self._compute_seq(new_seq, level-1)

    def reset(self) -> None:
        self.X = []
        self.Y = []
        self.number_call = 0


class Diagonal_Curve(Curve):
    def __init__(self, length: int = 4**4) -> None:
        """
        Args:
            length (int): the minimum length of the curve (length<4**level)
        """
        super().__init__("Diagonal", length, 2)

    def build_curve(self, level: int) -> None:
        X, Y = [0], [0]
        side_length = 2**level
        # iterates over the whole sequence
        x, y = 0, 0
        state = "NW"
        for i in range(side_length**2-1):
            x, y, state = self._consecutive_position(x, y, state, side_length)

            X.append(x)
            Y.append(y)

        self.X = X
        self.Y = Y

    def _consecutive_position(self, x: int, y: int, state: str, side_length: int) -> Tuple[int, int, str]:
        if state == "SE":
            if x == side_length-1:
                return x, y+1, "NW"
            if y == 0:
                return x+1, y, "NW"
            return x+1, y-1, "SE"
        if state == "NW":
            if x == 0:
                return x, y+1, "SE"
            if y == side_length-1:
                return x+1, y, "SE"
            return x-1, y+1, "NW"

    def reset(self) -> None:
        self.X = []
        self.Y = []


class Scan_Curve(Curve):
    def __init__(self, length: int = 4**4) -> None:
        """
        Args:
            length (int): the minimum length of the curve (length<4**level)
        """
        super().__init__("Scan", length, 2)

    def build_curve(self, level: int) -> None:
        X, Y = [0], [0]
        side_length = 2**level
        # iterates over the whole sequence
        x, y = 0, 0
        state = "N"
        for i in range(side_length**2-1):
            x, y, state = self._consecutive_position(x, y, state, side_length)

            X.append(x)
            Y.append(y)

        self.X = X
        self.Y = Y

    def _consecutive_position(self, x: int, y: int, state: str, side_length: int) -> Tuple[int, int, str]:
        if state == "N":
            if y == side_length-1:
                return x+1, y, "S"
            return x, y+1, "N"
        if state == "S":
            if y == 0:
                return x+1, y, "N"
            return x, y-1, "S"

    def reset(self) -> None:
        self.X = []
        self.Y = []


class Sweep_Curve(Curve):
    def __init__(self, length: int = 4**4) -> None:
        """
        Args:
            length (int): the minimum length of the curve (length<4**level)
        """
        super().__init__("Sweep", length, 2)

    def build_curve(self, level: int) -> None:
        X, Y = [0], [0]
        side_length = 2**level
        # iterates over the whole sequence
        x, y = 0, 0
        state = "N"
        for i in range(side_length**2-1):
            x, y, state = self._consecutive_position(x, y, state, side_length)

            X.append(x)
            Y.append(y)

        self.X = X
        self.Y = Y

    def _consecutive_position(self, x: int, y: int, state: str, side_length: int) -> Tuple[int, int, str]:
        if state == "N":
            if y == side_length-1:
                return x+1, 0, "N"
            return x, y+1, "N"

    def reset(self) -> None:
        self.X = []
        self.Y = []
