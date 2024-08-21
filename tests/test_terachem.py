import pytest
from qcio import CalcType

from qcparse.encoders.terachem import PADDING, XYZ_FILENAME, encode
from qcparse.exceptions import EncoderError, MatchNotFoundError
from qcparse.parsers.terachem import (
    calculation_succeeded,
    parse_calctype,
    parse_energy,
    parse_gradient,
    parse_hessian,
    parse_natoms,
    parse_nmo,
    parse_version_control_details,
    parse_version_string,
    parse_meci_energies,
)

from .data import gradients, hessians


@pytest.mark.parametrize(
    "filename,energy",
    (
        ("water.energy.out", -76.3861099088),
        ("water.gradient.out", -76.3861099088),
        ("water.frequencies.out", -76.3861099088),
    ),
)
def test_parse_energy(test_data_dir, data_collector, filename, energy):
    with open(test_data_dir / filename) as f:
        tcout = f.read()
    parse_energy(tcout, data_collector)


def test_parse_energy_positive(data_collector):
    energy = 76.3854579982
    parse_energy(f"FINAL ENERGY: {energy} a.u", data_collector)
    assert data_collector.energy == energy


@pytest.mark.parametrize(
    "energy",
    (-7634, 7123),
)
def test_parse_energy_integer(data_collector, energy):
    parse_energy(f"FINAL ENERGY: {energy} a.u", data_collector)
    assert data_collector.energy == energy


def test_parse_energy_raises_exception(data_collector):
    with pytest.raises(MatchNotFoundError):
        parse_energy("No energy here", data_collector)


@pytest.mark.parametrize(
    "filename,calctype",
    (
        ("water.energy.out", CalcType.energy),
        ("water.gradient.out", CalcType.gradient),
        ("water.frequencies.out", CalcType.hessian),
    ),
)
def test_parse_calctype(test_data_dir, filename, calctype):
    with open(test_data_dir / filename) as f:
        string = f.read()
    assert parse_calctype(string) == calctype


def test_parse_calctype_raises_exception():
    with pytest.raises(MatchNotFoundError):
        parse_calctype("No driver here")


def test_parse_version_git(terachem_energy_stdout):
    parsed = parse_version_string(terachem_energy_stdout)
    assert parsed == "v1.9-2022.03-dev [4daa16dd21e78d64be5415f7663c3d7c2785203c]"


def test_parse_version_hg(test_data_dir):
    hg_stdout = (test_data_dir / "hg.out").read_text()
    parsed = parse_version_string(hg_stdout)
    assert parsed == "v1.5K [ccdev]"


def test_calculation_succeeded(terachem_energy_stdout):
    assert calculation_succeeded(terachem_energy_stdout) is True
    assert (
        calculation_succeeded(
            """
        Incorrect purify value
        DIE called at line number 3572 in file terachem/params.cpp
         Job terminated: Thu Mar 23 03:47:12 2023
        """
        )
        is False
    )


@pytest.mark.parametrize(
    "filename,result",
    (
        ("failure.nocuda.out", False),
        ("failure.basis.out", False),
    ),
)
def test_calculation_succeeded_cuda_failure(test_data_dir, filename, result):
    with open(test_data_dir / filename) as f:
        tcout = f.read()
    assert calculation_succeeded(tcout) is result


@pytest.mark.parametrize(
    "filename,gradient",
    (
        (
            "water.gradient.out",
            gradients.water,
        ),
        (
            "caffeine.gradient.out",
            gradients.caffeine,
        ),
        (
            "caffeine.frequencies.out",
            gradients.caffeine_frequencies,
        ),
    ),
)
def test_parse_gradient(test_data_dir, filename, gradient, data_collector):
    with open(test_data_dir / filename) as f:
        tcout = f.read()

    parse_gradient(tcout, data_collector)
    assert data_collector.gradient == gradient


@pytest.mark.parametrize(
    "filename,hessian",
    (
        (
            "water.frequencies.out",
            hessians.water,
        ),
        (
            "caffeine.frequencies.out",
            hessians.caffeine,
        ),
    ),
)
def test_parse_hessian(test_data_dir, filename, hessian, data_collector):
    with open(test_data_dir / filename) as f:
        tcout = f.read()

    parse_hessian(tcout, data_collector)
    assert data_collector.hessian == hessian


@pytest.mark.parametrize(
    "filename,n_atoms",
    (("water.energy.out", 3), ("caffeine.gradient.out", 24)),
)
def test_parse_natoms(test_data_dir, filename, n_atoms, data_collector):
    with open(test_data_dir / filename) as f:
        tcout = f.read()

    parse_natoms(tcout, data_collector)
    assert data_collector.calcinfo_natoms == n_atoms


@pytest.mark.parametrize(
    "filename,nmo",
    (("water.energy.out", 13), ("caffeine.gradient.out", 146)),
)
def test_parse_nmo(test_data_dir, filename, nmo, data_collector):
    with open(test_data_dir / filename) as f:
        tcout = f.read()

    parse_nmo(tcout, data_collector)
    assert data_collector.calcinfo_nmo == nmo


def test_parse_git_commit(terachem_energy_stdout):
    git_commit = parse_version_control_details(terachem_energy_stdout)
    assert (
        git_commit
        == "4daa16dd21e78d64be5415f7663c3d7c2785203c"  # pragma: allowlist secret
    )


def test_write_input_files(prog_inp):
    """Test write_input_files method."""
    prog_inp = prog_inp("energy")

    native_input = encode(prog_inp)
    # Testing that we capture:
    # 1. Driver
    # 2. Structure
    # 3. Model
    # 4. Keywords (test booleans to lower case, ints, sts, floats)

    correct_tcin = (
        f"{'run':<{PADDING}} {prog_inp.calctype}\n"
        f"{'coordinates':<{PADDING}} {XYZ_FILENAME}\n"
        f"{'charge':<{PADDING}} {prog_inp.structure.charge}\n"
        f"{'spinmult':<{PADDING}} {prog_inp.structure.multiplicity}\n"
        f"{'method':<{PADDING}} {prog_inp.model.method}\n"
        f"{'basis':<{PADDING}} {prog_inp.model.basis}\n"
        f"{'purify':<{PADDING}} {prog_inp.keywords['purify']}\n"
        f"{'some-bool':<{PADDING}} "
        f"{str(prog_inp.keywords['some-bool']).lower()}\n"
    )
    assert native_input.input_file == correct_tcin


def test_write_input_files_renames_hessian_to_frequencies(prog_inp):
    """Test write_input_files method for hessian."""
    # Modify input to be a hessian calculation
    prog_inp = prog_inp("hessian")
    native_input = encode(prog_inp)

    assert native_input.input_file == (
        f"{'run':<{PADDING}} frequencies\n"
        f"{'coordinates':<{PADDING}} {XYZ_FILENAME}\n"
        f"{'charge':<{PADDING}} {prog_inp.structure.charge}\n"
        f"{'spinmult':<{PADDING}} {prog_inp.structure.multiplicity}\n"
        f"{'method':<{PADDING}} {prog_inp.model.method}\n"
        f"{'basis':<{PADDING}} {prog_inp.model.basis}\n"
        f"{'purify':<{PADDING}} {prog_inp.keywords['purify']}\n"
        f"{'some-bool':<{PADDING}} "
        f"{str(prog_inp.keywords['some-bool']).lower()}\n"
    )


def test_write_input_files_renames_meci_to_conical(prog_inp):
    """Test write_input_files method for hessian."""
    # Modify input to be a hessian calculation
    prog_inp = prog_inp("meci")
    native_input = encode(prog_inp)

    assert native_input.input_file == (
        f"{'run':<{PADDING}} conical\n"
        f"{'coordinates':<{PADDING}} {XYZ_FILENAME}\n"
        f"{'charge':<{PADDING}} {prog_inp.structure.charge}\n"
        f"{'spinmult':<{PADDING}} {prog_inp.structure.multiplicity}\n"
        f"{'method':<{PADDING}} {prog_inp.model.method}\n"
        f"{'basis':<{PADDING}} {prog_inp.model.basis}\n"
        f"{'purify':<{PADDING}} {prog_inp.keywords['purify']}\n"
        f"{'some-bool':<{PADDING}} "
        f"{str(prog_inp.keywords['some-bool']).lower()}\n"
    )


def test_encode_raises_error_qcio_args_passes_as_keywords(prog_inp):
    """These keywords should not be in the .keywords dict. They belong on structured
    qcio objects instead."""
    qcio_keywords_from_terachem = ["charge", "spinmult", "method", "basis", "run"]
    prog_inp = prog_inp("energy")
    for keyword in qcio_keywords_from_terachem:
        prog_inp.keywords[keyword] = "some value"
        with pytest.raises(EncoderError):
            encode(prog_inp)


def test_parse_meci_energies():
    """Test parsing of MECI energies from TeraChem output"""
    with open("tests/data/tc_meci.out", "r") as f:
        tc_output = f.read()

    lower_energies, upper_energies = parse_meci_energies(tc_output)

    real_lower_energies = [-76.41234567, -76.31234567]
    real_lower_energies = [
        -78.3602352844,
        -78.3613795195,
        -78.1972612321,
        -78.1230911484,
        -78.1435640726,
        -78.1581173469,
        -78.1743452893,
        -78.1767563570,
        -78.1808970467,
        -78.1849048022,
        -78.1932607853,
        -78.2109414871,
        -78.2109414871,
        -78.2115357606,
        -78.2131667033,
        -78.2193499936,
        -78.2200701178,
        -78.2221773826,
        -78.2231580283,
        -78.2241718470,
        -78.2257009112,
        -78.2285351005,
        -78.2285351005,
        -78.2291643471,
        -78.2312002461,
        -78.2382912379,
        -78.2416133622,
        -78.2447895284,
        -78.2507436588,
        -78.2568010565,
        -78.2609666302,
        -78.2631034963,
        -78.2658904891,
        -78.2714451592,
        -78.2768670995,
        -78.2804075126,
        -78.2854013229,
        -78.2879234706,
        -78.2888912345,
        -78.2901950460,
        -78.2909614826,
        -78.2923798602,
        -78.2936528771,
        -78.2943897524,
        -78.2946579620,
        -78.2947641869,
        -78.2948082206,
        -78.2948377739,
        -78.2948756100,
        -78.2949238394,
        -78.2949753007,
        -78.2950070987,
        -78.2950582294,
        -78.2950784604,
        -78.2952177468,
        -78.2952114442,
        -78.2958741133,
        -78.2945536503,
        -78.2952219133,
        -78.2952320183,
        -78.2952509141,
        -78.2952575624,
        -78.2952905456,
        -78.2952885794,
        -78.2952382818,
        -78.2952879962,
        -78.2953008594,
        -78.2953092232,
        -78.2953110190,
        -78.2953131343,
        -78.2953236017,
        -78.2953277915,
        -78.2953333062,
        -78.2953459633,
        -78.2953435395,
        -78.2953448131,
        -78.2953458330,
        -78.2953459124,
        -78.2953461700,
        -78.2953467876,
        -78.2953495077,
        -78.2953490716,
        -78.2953491383,
        -78.2953497274,
        -78.2953496345
    ]
    real_upper_energies = [
        -78.1916020489,
        -78.1922673349,
        -78.1451546808,
        -78.1270053582,
        -78.1434325632,
        -78.1573304656,
        -78.1733826157,
        -78.1768116425,
        -78.1812605878,
        -78.1849783706,
        -78.1937404123,
        -78.2114767948,
        -78.2114766178,
        -78.2119131084,
        -78.2133273964,
        -78.2193098486,
        -78.2200726632,
        -78.2222045744,
        -78.2231418433,
        -78.2241185587,
        -78.2255517278,
        -78.2280245328,
        -78.2280245214,
        -78.2284586131,
        -78.2313672646,
        -78.2380233211,
        -78.2414027929,
        -78.2446235228,
        -78.2504950268,
        -78.2569156772,
        -78.2610594752,
        -78.2630758482,
        -78.2658485136,
        -78.2713762612,
        -78.2768535014,
        -78.2804098348,
        -78.2854309712,
        -78.2878230654,
        -78.2889144763,
        -78.2902061084,
        -78.2910062728,
        -78.2926856680,
        -78.2938825084,
        -78.2944768676,
        -78.2946679100,
        -78.2947649999,
        -78.2948075260,
        -78.2948374958,
        -78.2948761397,
        -78.2949263685,
        -78.2949789295,
        -78.2950067960,
        -78.2950559507,
        -78.2950787627,
        -78.2951780220,
        -78.2952117992,
        -78.2952320655,
        -78.2951900144,
        -78.2952205901,
        -78.2952320521,
        -78.2952505842,
        -78.2952573235,
        -78.2952830461,
        -78.2952886721,
        -78.2952158849,
        -78.2952880771,
        -78.2952970086,
        -78.2953084674,
        -78.2953110484,
        -78.2953131541,
        -78.2953250451,
        -78.2953279210,
        -78.2953331560,
        -78.2953427055,
        -78.2953435847,
        -78.2953442660,
        -78.2953450577,
        -78.2953453413,
        -78.2953462152,
        -78.2953468311,
        -78.2953487147,
        -78.2953489087,
        -78.2953490826,
        -78.2953494591,
        -78.2953495520
    ]

    # Check if the energies are floats and in a reasonable range
    for lower, upper, real_lower, real_upper in zip(
        lower_energies, upper_energies, real_lower_energies, real_upper_energies
    ):
        assert lower == real_lower
        assert upper == real_upper
