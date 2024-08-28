"""Parsers for TeraChem output files."""

import re
from typing import Union, List, Tuple
from pathlib import Path

from qcio import CalcType, Structure, OptimizationResults

from qcparse.exceptions import MatchNotFoundError
from qcparse.models import FileType, ParsedDataCollector

from .utils import parser, regex_search

SUPPORTED_FILETYPES = {FileType.stdout}


def parse_calctype(string: str) -> CalcType:
    """Parse the calctype from TeraChem stdout."""
    calctypes = {
        r"RUNNING GEOMETRY OPTMIZATION": CalcType.minimize,
        r"SINGLE POINT NONADIABATIC COUPLING": CalcType.coupling,
        r"SEARCHING FOR THE TRANSITION STATE": CalcType.neb,
        r"SINGLE POINT ENERGY CALCULATIONS": CalcType.energy,
        r"SINGLE POINT GRADIENT CALCULATIONS": CalcType.gradient,
        r"FREQUENCY ANALYSIS": CalcType.hessian,
    }

    for regex, calctype in calctypes.items():
        match = re.search(regex, string)
        if match:
            return calctype
    raise MatchNotFoundError(regex, string)


@parser()
def parse_energy(string: str, data_collector: ParsedDataCollector):
    """Parse the final energy from TeraChem stdout.

    NOTE:
        - Works on frequency files containing many energy values because re.search()
            returns the first result
    """
    regex = r"FINAL ENERGY: (-?\d+(?:\.\d+)?)"
    data_collector.energy = float(regex_search(regex, string).group(1))


@parser(only=[CalcType.energy])
def parse_energy_subtype(string: str, data_collector: ParsedDataCollector):
    """Parse the energy subtype from TeraChem stdout."""
    energy_subtypes = {
        r"EOM-CCSD Energies": CalcType.energy_eom_ccsd,
        r"Restricted CIS Parameters": CalcType.energy_cis,
        r"Restricted hh-TDA Parameters": CalcType.energy_hhtda,
        r"Active Space Parameters": CalcType.energy_cas,
    }
    for regex, calctype in energy_subtypes.items():
        match = re.search(regex, string)
        if match:
            data_collector.energy_subtype = calctype


@parser(only=[CalcType.energy_cis])
def parse_cis_numstates(string: str, data_collector: ParsedDataCollector):
    """Parse the number of states from TeraChem stdout."""
    regex = r"Number of roots to find:\s*(\d+)"
    data_collector.cis_info_numstates = int(regex_search(regex, string).group(1))


@parser(only=[CalcType.energy_cis])
def parse_cis_energies(string: str, data_collector: ParsedDataCollector):
    """Parse the cis energies from TeraChem stdout."""
    regex = r"^\s*\d+\s+(-?\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+-?\d+\.\d+\s+\d+\s*->\s*\d"
    matches = re.findall(regex, string, re.MULTILINE)

    if matches:
        data_collector.cis_energies = [float(energy) for energy in matches]
    else:
        raise MatchNotFoundError(regex, string)


def parse_eom_ccsd_numstates(string: str, data_collector: ParsedDataCollector):
    """Parse the number of states from TeraChem stdout."""
    regex = r"Number of states:\s*(\d+)"
    data_collector.eom_ccsd_numstates = int(regex_search(regex, string).group(1))


@parser(only=[CalcType.energy_eom_ccsd])
def parse_eom_ccsd_energies(string: str, data_collector: ParsedDataCollector):
    regex = r"^\s*Root\s+\d+:\s+(-?\d+\.\d+)"  # TODO: Find a better regex to get rid of silly logic below
    matches = re.findall(regex, string, re.MULTILINE)

    assert data_collector.eom_ccsd_numstates
    # Here eom_ccsd_numstates only counts number of excited states
    # we need to include the ground state in the number of states
    matches = matches[-(data_collector.eom_ccsd_numstates + 1) :]
    if matches:
        data_collector.eom_ccsd_energies = [float(energy) for energy in matches]
    else:
        raise MatchNotFoundError(regex, string)


@parser(only=[CalcType.gradient, CalcType.hessian])
def parse_gradient(string: str, data_collector: ParsedDataCollector):
    """Parse gradient from TeraChem stdout."""
    # This will match all floats after the dE/dX dE/dY dE/dZ header and stop at the
    # terminating ---- line
    regex = r"(?<=dE\/dX\s{12}dE\/dY\s{12}dE\/dZ\n)[\d\.\-\s]+(?=\n-{2,})"
    gradient_string = regex_search(regex, string).group()

    # split string and cast to floats
    values = [float(val) for val in gradient_string.split()]

    # arrange into N x 3 gradient
    gradient = []
    for i in range(0, len(values), 3):
        gradient.append(values[i : i + 3])

    data_collector.gradient = gradient


@parser(only=[CalcType.hessian])
def parse_hessian(string: str, data_collector: ParsedDataCollector):
    """Parse Hessian Matrix from TeraChem stdout

    Notes:
        This function searches the entire document N times for all regex matches where
        N is the number of atoms. This makes the function's code easy to reason about.
        If performance becomes an issues for VERY large Hessians (unlikely) you can
        accelerate this function by parsing all Hessian floats in one pass, like the
        parse_gradient function above, and then doing the math to figure out how to
        properly sequence those values to from the Hessian matrix given TeraChem's
        six-column format for printing out Hessian matrix entries.
    """
    # requires .format(int). {{}} values are to escape {15|2} for .format()
    regex = r"(?:\s+{}\s)((?:\s-?\d\.\d{{15}}e[+-]\d{{2}})+)"
    hessian = []

    # Match all rows containing Hessian data; one set of rows at a time
    count = 1
    while matches := re.findall(regex.format(count), string):
        row = []
        for match in matches:
            row.extend([float(val) for val in match.split()])
        hessian.append(row)
        count += 1

    if not hessian:
        raise MatchNotFoundError(regex, string)

    # Assert we have created a square Hessian matrix
    for i, row in enumerate(hessian):
        assert len(row) == len(
            hessian
        ), "We must have missed some floats. Hessian should be a square matrix. Only "
        f"recovered {len(row)} of {len(hessian)} floats for row {i}."

    data_collector.hessian = hessian


@parser()
def parse_natoms(string: str, data_collector: ParsedDataCollector):
    """Parse number of atoms value from TeraChem stdout"""
    regex = r"Total atoms:\s*(\d+)"
    data_collector.calcinfo_natoms = int(regex_search(regex, string).group(1))


@parser()
def parse_nmo(string: str, data_collector: ParsedDataCollector):
    """Parse the number of molecular orbitals TeraChem stdout"""
    regex = r"Total orbitals:\s*(\d+)"
    data_collector.calcinfo_nmo = int(regex_search(regex, string).group(1))


def parse_version_control_details(string: str) -> str:
    """Parse TeraChem git commit or Hg version from TeraChem stdout."""
    regex = r"(Git|Hg) Version: (\S*)"
    return regex_search(regex, string).group(2)


def parse_terachem_version(string: str) -> str:
    """Parse TeraChem version from TeraChem stdout."""
    regex = r"TeraChem (v\S*)"
    return regex_search(regex, string).group(1)


def parse_version_string(string: str) -> str:
    """Parse version string plus git commit from TeraChem stdout.

    Matches format of 'terachem --version' on command line.
    """
    return f"{parse_terachem_version(string)} [{parse_version_control_details(string)}]"


def parse_meci_energies(
    string: str, data_collector: ParsedDataCollector
) -> Tuple[List[float], List[float]]:
    """Parse lower and upper state energies from TeraChem MECI calculation stdout"""
    lower_regex = r"Lower state energy:\s*(-?\d+\.\d+)"
    upper_regex = r"Upper state energy:\s*(-?\d+\.\d+)"

    lower_matches = re.findall(lower_regex, string)
    upper_matches = re.findall(upper_regex, string)

    if not lower_matches or not upper_matches:
        raise MatchNotFoundError("Lower or upper state energy", string)

    # Convert matches to floats and store in data_collector
    lower_state_energies = [float(energy) for energy in lower_matches]
    upper_state_energies = [float(energy) for energy in upper_matches]

    data_collector.lower_state_energies = lower_state_energies
    data_collector.upper_state_energies = upper_state_energies
    return data_collector.lower_state_energies, data_collector.upper_state_energies


def calculation_succeeded(string: str) -> bool:
    """Determine from TeraChem stdout if a calculation completed successfully."""
    regex = r"Job finished:"
    if re.search(regex, string):
        # If any match for a failure regex is found, the calculation failed
        return True
    return False


def parse_meci_dir(directory: Union[str, Path]) -> OptimizationResults:
    """Parse the output of a TeraChem meci calculation.

    Args:
        directory: Directory containing the output of a TeraChem meci calculation.

    Returns:
        An OptimizationResults object containing the results of the calculation.
    """
    # Parses all the structures in the meci_conformers.xyz file
    geoms: List[Structure] = Structure.open(directory / "meci_conformers.xyz")
    # Comment values are at struct.extras[Structure._xyz_comment_key]

    # Parse the output file
    with open(directory / "meci.out", "r") as f:
        string = f.read()

    # Parse the energy
    regex = r"FINAL ENERGY: (-?\d+(?:\.\d+)?)"
    energy = float(regex_search(regex, string).group(1))

    return energy


def parse_neb(string: str, data_collector: ParsedDataCollector): ...


def parse_cas(string: str, data_collector: ParsedDataCollector): ...


def parse_hhtda(string: str, data_collector: ParsedDataCollector): ...


def parse_md(string: str, data_collector: ParsedDataCollector): ...


def parse_minimize(string: str, data_collector: ParsedDataCollector): ...


def parse_coupling(string: str, data_collector: ParsedDataCollector): ...
