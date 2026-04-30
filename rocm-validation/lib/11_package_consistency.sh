#!/usr/bin/env bash
# Module 11 — Package Consistency

test_package_consistency() {
  section "11 · PACKAGE CONSISTENCY"

  capture "rocm_packages" bash -c \
    "dpkg -l 2>/dev/null | grep -i rocm | sort || \
     rpm -qa 2>/dev/null | grep -i rocm | sort || \
     echo 'no pkg manager found'"

  if cmd_exists dpkg; then
    # All ROCm packages should be same major.minor
    local versions
    versions=$(dpkg -l 2>/dev/null | grep -i rocm | awk '{print $3}' | \
               grep -oP "^[0-9]+\.[0-9]+" | sort -u || true)
    local ver_count
    ver_count=$(echo "$versions" | grep -c . | tr -d '[:space:]')
    ver_count=$(_int "$ver_count")

    if [[ "$ver_count" -le 1 ]]; then
      record_pass "no_mixed_rocm_packages" "All ROCm packages on same version branch"
    else
      record_fail "no_mixed_rocm_packages" \
        "Multiple ROCm version branches: $(echo "$versions" | tr '\n' ' ')"
    fi

    local rocm_major_expected
    rocm_major_expected=$(echo "$ROCM_EXPECTED_VERSION" | cut -d. -f1,2)
    local wrong_ver
    wrong_ver=$(dpkg -l 2>/dev/null | grep -i rocm | awk '{print $3}' | \
                grep -vP "^${rocm_major_expected//./\\.}" | head -5 || true)
    if [[ -z "$wrong_ver" ]]; then
      record_pass "rocm_package_versions_correct" "All packages on $rocm_major_expected.x"
    else
      record_fail "rocm_package_versions_correct" \
        "Packages on wrong version: $(echo "$wrong_ver" | tr '\n' ' ')"
    fi

    local pkg_count
    pkg_count=$(dpkg -l 2>/dev/null | grep -c "^ii.*rocm" | tr -d '[:space:]')
    pkg_count=$(_int "$pkg_count")
    record_pass "rocm_package_count" "$pkg_count ROCm packages installed"

    local broken
    broken=$(dpkg -l 2>/dev/null | grep -E "^iH|^iF|^iU" | grep -c rocm | tr -d '[:space:]')
    broken=$(_int "$broken")
    if [[ "$broken" -eq 0 ]]; then
      record_pass "no_broken_packages"
    else
      record_fail "no_broken_packages" "$broken partially-installed ROCm package(s)"
    fi

    for pkg in rocm-dev rocm-libs hip-base hipcc; do
      if dpkg -l 2>/dev/null | grep -q "^ii.*$pkg"; then
        record_pass "pkg_${pkg//-/_}" "$pkg installed"
      else
        record_warn "pkg_${pkg//-/_}" "$pkg not found — may be expected"
      fi
    done
  else
    for t in no_mixed_rocm_packages rocm_package_versions_correct \
              rocm_package_count no_broken_packages; do
      record_skip "$t" "dpkg not available"
    done
  fi

  local version_file
  version_file=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | tr -d '[:space:]' || echo "")
  if [[ -n "$version_file" ]]; then
    record_pass "rocm_version_file" "/opt/rocm/.info/version = $version_file"
  else
    record_warn "rocm_version_file" "/opt/rocm/.info/version not found"
  fi

  if python3 -c "import torch; assert torch.cuda.is_available() or hasattr(torch,'hip')" 2>/dev/null; then
    local torch_ver hip_ver
    torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    hip_ver=$(python3 -c "import torch; print(getattr(torch.version,'hip','N/A'))" 2>/dev/null || echo "N/A")
    record_pass "pytorch_rocm" "PyTorch $torch_ver (HIP $hip_ver)"
  else
    record_skip "pytorch_rocm" "PyTorch not installed (optional)"
  fi
}
