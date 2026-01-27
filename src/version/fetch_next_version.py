from version import Version, parse_args


def main():
    args = parse_args()
    version = Version(args.input_version_datastore)
    print(version.fetch_next_version())

if __name__ == '__main__':
    main()