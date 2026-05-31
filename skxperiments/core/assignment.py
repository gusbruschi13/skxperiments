"""Assignment classes representing the contract between designs and estimators.

An Assignment object is always created by a Design via randomize() — never
instantiated directly by the user. It carries the treatment assignment
alongside a copy of the original data, ensuring no side effects.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from skxperiments.core.exceptions import InvalidDesignError


class BaseAssignment(ABC):
    """Abstract base class for all assignment objects.

    An Assignment is the contract between a design and an estimator.
    Estimators receive Assignment objects, not loose DataFrames.

    Notes
    -----
    Assignment objects are intended to be immutable after construction.
    Designs are responsible for passing a defensive copy of the input
    DataFrame (i.e., ``df.copy()``) when constructing an Assignment.
    The outcome variable is **not** part of the Assignment contract:
    estimators receive the outcome column name as a parameter at __init__
    and resolve it against ``assignment.data_`` at fit time.

    Parameters
    ----------
    data : pd.DataFrame
        Copy of the original DataFrame with treatment column added.
    treatment_col : str
        Name of the treatment column.
    design : Any
        Reference to the design that generated this assignment.

    Attributes
    ----------
    data_ : pd.DataFrame
        DataFrame with treatment column added.
    treatment_col_ : str
        Name of the treatment column.
    design_ : Any
        Reference to the generating design.
    n_units_ : int
        Total number of units.
    n_treated_ : int
        Number of units assigned to treatment.
    n_control_ : int
        Number of units assigned to control.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        design: Any,
    ) -> None:
        if treatment_col not in data.columns:
            raise InvalidDesignError(
                f"Treatment column '{treatment_col}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}."
            )
    
        self.data_: pd.DataFrame = data
        self.treatment_col_: str = treatment_col
        self.design_: Any = design
        self.n_units_: int = len(data)
        self.n_treated_: int = int((data[treatment_col] == 1).sum())
        self.n_control_: int = int((data[treatment_col] == 0).sum())

    def _validate_treatment_col(self) -> None:
        """Validate that the treatment column contains only 0 and 1.

        Raises
        ------
        InvalidDesignError
            If the treatment column contains values other than 0 and 1.
        """
        unique_values = set(self.data_[self.treatment_col_].unique())
        allowed = {0, 1}
        if not unique_values.issubset(allowed):
            invalid = unique_values - allowed
            raise InvalidDesignError(
                f"Treatment column '{self.treatment_col_}' must contain only "
                f"0 and 1, but found values: {sorted(invalid)}."
            )

    @abstractmethod
    def treated_ids(self) -> np.ndarray:
        """Return iloc positions of treated units.

        Returns
        -------
        np.ndarray
            Array of integer positions where treatment == 1.
        """

    @abstractmethod
    def control_ids(self) -> np.ndarray:
        """Return iloc positions of control units.

        Returns
        -------
        np.ndarray
            Array of integer positions where treatment == 0.
        """

    @abstractmethod
    def draw(self, seed: int | None = None) -> "BaseAssignment":
        """Generate a new realization of the assignment under the same mechanism.

        This method enables randomization-based inference: each call returns
        a fresh Assignment object whose treatment vector is a new draw from
        the same randomization mechanism that produced this Assignment
        (same N, same probabilities, same constraints), but with a different
        random seed.

        The returned Assignment must be of the same concrete type as self
        and reference the same underlying data. Implementations must not
        mutate self.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the new draw. If None, a non-deterministic
            draw is performed.

        Returns
        -------
        BaseAssignment
            New Assignment of the same concrete type, with a freshly drawn
            treatment vector.

        Notes
        -----
        Consumed by RandomizationTest to generate the null distribution.
        Not intended for direct use by end users.
        """

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            Format: ClassName(n_treated=N, n_control=N)
        """
        class_name = type(self).__name__
        return f"{class_name}(n_treated={self.n_treated_}, n_control={self.n_control_})"


class CRDAssignment(BaseAssignment):
    """Assignment resulting from a Completely Randomized Design.

    Parameters
    ----------
    data : pd.DataFrame
        Copy of the original DataFrame with treatment column added.
    treatment_col : str
        Name of the treatment column.
    design : Any
        Reference to the design that generated this assignment.
    seed : int or None, optional
        Random seed used for the randomization, by default None.
    rerandomization_metadata : dict or None, optional
        When the assignment was produced by a rerandomization design
        (e.g., ReRandomizedCRD), this dict carries the information
        needed to replay the acceptance/rejection mechanism in
        ``draw()``. Expected keys:

        - ``"covariates"``: list[str]
        - ``"threshold"``: float
        - ``"cov_matrix"``: np.ndarray (sample covariance, ddof=1, of
          the full DataFrame)
        - ``"attempts"``: int
        - ``"scaling_factor"``: float (1/n_treated + 1/n_control)

        When None, this is a plain CRD assignment.

    Attributes
    ----------
    seed_ : int or None
        Seed used in the randomization.
    rerandomization_metadata : dict or None
        Rerandomization metadata, or None if this is a plain CRD
        assignment.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1, 2, 3, 4], "treatment": [1, 0, 1, 0]})
    >>> assignment = CRDAssignment(
    ...     data=df, treatment_col="treatment", design=None, seed=42
    ... )
    >>> assignment.n_treated_
    2
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        design: Any,
        seed: int | None = None,
        rerandomization_metadata: dict | None = None,
    ) -> None:
        super().__init__(data=data, treatment_col=treatment_col, design=design)
        self.seed_: int | None = seed
        self.rerandomization_metadata: dict | None = rerandomization_metadata
        self._validate_treatment_col()

    def treated_ids(self) -> np.ndarray:
        """Return iloc positions of treated units."""
        return np.where(self.data_[self.treatment_col_].values == 1)[0]

    def control_ids(self) -> np.ndarray:
        """Return iloc positions of control units."""
        return np.where(self.data_[self.treatment_col_].values == 0)[0]

    def draw(self, seed: int | None = None) -> "CRDAssignment":
        """Generate a new realization under the same mechanism.

        For plain CRD, delegates to ``design_.randomize(df_clean)``.
        For rerandomized CRD, delegates to
        ``design_._randomize_with_cached_cov`` so the covariance matrix
        is reused without recomputation.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the new draw, by default None.

        Returns
        -------
        CRDAssignment
            Fresh assignment under the same mechanism.

        Raises
        ------
        InvalidDesignError
            If ``design_`` is None.
        """
        if self.design_ is None:
            raise InvalidDesignError(
                "Cannot draw a new realization: this CRDAssignment was "
                "constructed without a reference to a generating design."
            )

        df_clean = self.data_.drop(columns=[self.treatment_col_])

        if self.rerandomization_metadata is not None:
            # Rerandomized CRD: reuse cached covariance matrix.
            cov_matrix = self.rerandomization_metadata["cov_matrix"]
            rng = np.random.default_rng(seed)
            return self.design_._randomize_with_cached_cov(
                df=df_clean,
                cov_matrix=cov_matrix,
                rng=rng,
            )

        # Plain CRD: delegate to design.randomize via temporary seed override.
        original_seed = getattr(self.design_, "seed", None)
        try:
            self.design_.seed = seed
            return self.design_.randomize(df_clean)
        finally:
            self.design_.seed = original_seed


class BlockedAssignment(BaseAssignment):
    """Assignment resulting from a Blocked Completely Randomized Design.

    Treatment is randomized independently within each block, preserving
    the treatment proportion within every block.

    Parameters
    ----------
    data : pd.DataFrame
        Copy of the original DataFrame with treatment column added.
    treatment_col : str
        Name of the treatment column.
    design : Any
        Reference to the BlockedCRD design that generated this assignment.
    block_col : str
        Name of the column identifying blocks.
    block_sizes : dict
        Mapping from block label to number of units in that block.
    seed : int or None, optional
        Random seed used for the randomization, by default None.

    Attributes
    ----------
    block_col_ : str
        Name of the block column.
    block_sizes_ : dict
        Mapping from block label to block size.
    seed_ : int or None
        Seed used in the randomization.

    Notes
    -----
    The block column must exist in ``data``. The treatment proportion
    is preserved within each block; see BlockedCRD for the design that
    produces this Assignment.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        design: Any,
        block_col: str,
        block_sizes: dict,
        seed: int | None = None,
    ) -> None:
        super().__init__(data=data, treatment_col=treatment_col, design=design)
        if block_col not in data.columns:
            raise InvalidDesignError(
                f"Block column '{block_col}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}."
            )
        self.block_col_: str = block_col
        self.block_sizes_: dict = dict(block_sizes)
        self.seed_: int | None = seed
        self._validate_treatment_col()

    def treated_ids(self) -> np.ndarray:
        """Return iloc positions of treated units."""
        return np.where(self.data_[self.treatment_col_].values == 1)[0]

    def control_ids(self) -> np.ndarray:
        """Return iloc positions of control units."""
        return np.where(self.data_[self.treatment_col_].values == 0)[0]

    def draw(self, seed: int | None = None) -> "BlockedAssignment":
        """Generate a new realization under the same blocked mechanism.

        Delegates to the generating design, ensuring the same block
        structure and within-block proportion are respected.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the new draw.

        Returns
        -------
        BlockedAssignment
            New BlockedAssignment with a freshly drawn treatment vector.

        Raises
        ------
        InvalidDesignError
            If this Assignment was constructed without a reference to
            a generating design.
        """
        if self.design_ is None:
            raise InvalidDesignError(
                "Cannot draw a new realization: this BlockedAssignment "
                "was constructed without a reference to a generating "
                "design."
            )

        df_clean = self.data_.drop(columns=[self.treatment_col_])

        original_seed = getattr(self.design_, "seed", None)
        try:
            self.design_.seed = seed
            return self.design_.randomize(df_clean)
        finally:
            self.design_.seed = original_seed


class FactorialAssignment(BaseAssignment):
    """Assignment resulting from a 2^K Factorial Design.

    Each unit is assigned to one of 2^K cells defined by the values of
    K binary factors. The synthetic column ``"_cell"`` carries an
    integer in [0, 2^K - 1] encoding the cell of each unit.

    Cell encoding convention (little-endian):

        cell_index = sum(factor_value * 2**i
                         for i, factor_value in enumerate(factor_cols))

    For K=2 with factors ``["A", "B"]``:
        A=0, B=0 -> cell 0
        A=1, B=0 -> cell 1
        A=0, B=1 -> cell 2
        A=1, B=1 -> cell 3

    This convention is part of the contract: FactorialEstimator
    (Phase 3) relies on it to reconstruct factor effects from cell
    indices.

    Notes
    -----
    Unlike CRDAssignment and BlockedAssignment, FactorialAssignment
    does **not** call ``self._validate_treatment_col()`` in __init__.
    The synthetic ``"_cell"`` column carries values in
    [0, 2^K - 1], not a binary 0/1 treatment indicator. Validating it
    against {0, 1} would always fail for K >= 1.

    Consequently, ``treated_ids()`` and ``control_ids()`` are not
    meaningful for factorial designs and raise NotImplementedError.
    Use ``cell_ids(**factor_values)`` to select units by cell.

    Parameters
    ----------
    data : pd.DataFrame
        Copy of the original DataFrame with factor columns and
        ``"_cell"`` added.
    design : Any
        Reference to the FactorialDesign that generated this assignment.
    factor_cols : list of str
        Names of the K factors, in the order used to compute cell indices.
    cell_sizes : dict
        Mapping from cell index to number of units in that cell.
    seed : int or None, optional
        Random seed used in the randomization, by default None.

    Attributes
    ----------
    factor_cols : list of str
        Names of the K factors.
    n_cells_ : int
        Number of cells (2^K).
    cell_sizes_ : dict
        Mapping from cell index to cell size.
    seed_ : int or None
        Seed used in the randomization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        design: Any,
        factor_cols: list[str],
        cell_sizes: dict,
        seed: int | None = None,
    ) -> None:
        # Pass "_cell" as treatment_col to satisfy BaseAssignment's
        # interface, but do NOT validate it against {0, 1}: see
        # class docstring.
        super().__init__(data=data, treatment_col="_cell", design=design)
        # Override the binary-coded n_treated_ / n_control_ set by
        # BaseAssignment; they are not meaningful here.
        self.n_treated_ = 0
        self.n_control_ = 0

        self.factor_cols: list[str] = list(factor_cols)
        self.n_cells_: int = 2 ** len(factor_cols)
        self.cell_sizes_: dict = dict(cell_sizes)
        self.seed_: int | None = seed

    def treated_ids(self) -> np.ndarray:
        """Not applicable to factorial designs.

        Raises
        ------
        NotImplementedError
            Always. Use ``cell_ids(**factor_values)`` instead.
        """
        # TODO Fase 3: revisar contrato de treated_ids/control_ids
        # quando FactorialEstimator for implementado — atual
        # NotImplementedError é solução de transição.
        raise NotImplementedError(
            "FactorialAssignment não tem tratamento binário único — "
            "use cell_ids(**factor_values) para selecionar unidades "
            "por célula"
        )

    def control_ids(self) -> np.ndarray:
        """Not applicable to factorial designs.

        Raises
        ------
        NotImplementedError
            Always. Use ``cell_ids(**factor_values)`` instead.
        """
        # TODO Fase 3: revisar contrato de treated_ids/control_ids
        # quando FactorialEstimator for implementado — atual
        # NotImplementedError é solução de transição.
        raise NotImplementedError(
            "FactorialAssignment não tem tratamento binário único — "
            "use cell_ids(**factor_values) para selecionar unidades "
            "por célula"
        )

    def cell_ids(self, **factor_values: int) -> np.ndarray:
        """Return iloc positions of units in the given cell.

        Parameters
        ----------
        **factor_values : int
            Mapping of factor name to its value (0 or 1). All factors
            in ``factor_cols`` must be specified.

        Returns
        -------
        np.ndarray
            Array of iloc positions of units matching all specified
            factor values.

        Raises
        ------
        InvalidDesignError
            If a kwarg is not a known factor name, if a value is not
            0 or 1, or if not all factors are specified.

        Examples
        --------
        >>> # For a 2x2 design with factors ["A", "B"]:
        >>> # assignment.cell_ids(A=1, B=0) -> iloc positions of cell 1
        """
        # Validate factor names
        unknown = [k for k in factor_values if k not in self.factor_cols]
        if unknown:
            raise InvalidDesignError(
                f"Unknown factor(s) in cell_ids: {unknown}. "
                f"Known factors: {self.factor_cols}."
            )

        # Validate all factors specified
        missing = [k for k in self.factor_cols if k not in factor_values]
        if missing:
            raise InvalidDesignError(
                f"cell_ids requires values for all factors. "
                f"Missing: {missing}. Provided: {list(factor_values.keys())}."
            )

        # Validate values
        for name, value in factor_values.items():
            if value not in (0, 1):
                raise InvalidDesignError(
                    f"Factor '{name}' value must be 0 or 1, "
                    f"received {value!r}."
                )

        # Build mask: all factors must match.
        mask = np.ones(self.n_units_, dtype=bool)
        for name in self.factor_cols:
            mask &= self.data_[name].values == factor_values[name]

        return np.where(mask)[0]

    def draw(self, seed: int | None = None) -> "FactorialAssignment":
        """Generate a new realization preserving cell sizes.

        Drops factor columns and ``"_cell"`` from the data before
        delegating to ``design_.randomize`` to avoid name-collision
        errors in randomize().

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the new draw, by default None.

        Returns
        -------
        FactorialAssignment
            New assignment with freshly drawn cell allocation.

        Raises
        ------
        InvalidDesignError
            If ``design_`` is None.
        """
        if self.design_ is None:
            raise InvalidDesignError(
                "Cannot draw a new realization: this FactorialAssignment "
                "was constructed without a reference to a generating "
                "design."
            )

        cols_to_drop = ["_cell"] + list(self.factor_cols)
        df_clean = self.data_.drop(columns=cols_to_drop)

        original_seed = getattr(self.design_, "seed", None)
        try:
            self.design_.seed = seed
            return self.design_.randomize(df_clean)
        finally:
            self.design_.seed = original_seed