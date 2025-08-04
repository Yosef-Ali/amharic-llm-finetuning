#!/bin/bash

# Run tests for Amharic H-Net model
#
# This script provides a convenient way to run the test suite for the Amharic H-Net model.
# It supports running all tests, specific test classes, or specific test methods.

set -e

# Default values
VERBOSE=false
FAIL_FAST=false
TEST_CLASS=""
TEST_METHOD=""
OUTPUT_FILE=""

# Function to display usage information
function display_usage {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Display this help message"
    echo "  -v, --verbose              Run tests in verbose mode"
    echo "  -f, --fail-fast            Stop on first test failure"
    echo "  -c, --class CLASS          Run tests for specific class"
    echo "  -m, --method METHOD        Run specific test method (requires --class)"
    echo "  -o, --output FILE          Save test results to file"
    echo ""
    echo "Examples:"
    echo "  $0                         Run all tests"
    echo "  $0 -v                      Run all tests in verbose mode"
    echo "  $0 -c TestAmharicPreprocessor  Run tests for TestAmharicPreprocessor class"
    echo "  $0 -c TestAmharicPreprocessor -m test_remove_non_amharic  Run specific test method"
    echo "  $0 -o test_results.txt     Save test results to file"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            display_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--fail-fast)
            FAIL_FAST=true
            shift
            ;;
        -c|--class)
            TEST_CLASS="$2"
            shift
            shift
            ;;
        -m|--method)
            TEST_METHOD="$2"
            shift
            shift
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            display_usage
            exit 1
            ;;
    esac
done

# Build the command
COMMAND="python -m unittest"

# Add verbose flag if requested
if [ "$VERBOSE" = true ]; then
    COMMAND="$COMMAND -v"
fi

# Add fail-fast flag if requested
if [ "$FAIL_FAST" = true ]; then
    COMMAND="$COMMAND -f"
fi

# Add test class and method if specified
if [ -n "$TEST_CLASS" ]; then
    if [ -n "$TEST_METHOD" ]; then
        COMMAND="$COMMAND test_suite.$TEST_CLASS.$TEST_METHOD"
    else
        COMMAND="$COMMAND test_suite.$TEST_CLASS"
    fi
else
    if [ -n "$TEST_METHOD" ]; then
        echo "Error: --method requires --class to be specified"
        display_usage
        exit 1
    else
        COMMAND="$COMMAND test_suite"
    fi
fi

# Run the command
echo "Running: $COMMAND"
if [ -n "$OUTPUT_FILE" ]; then
    echo "Saving results to: $OUTPUT_FILE"
    eval "$COMMAND" | tee "$OUTPUT_FILE"
else
    eval "$COMMAND"
fi

# Check if tests passed
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi