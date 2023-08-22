def get_user_input(subdirectories, default_choice=None):
    print("Select a subdirectory:")
    for index, subdirectory in enumerate(subdirectories):
        print(f"{index + 1}. {subdirectory}")
    default_index = default_choice if default_choice else 1
    choice = input(f"Enter your choice [{default_index}]: ")
    if choice:
        return subdirectories[int(choice) - 1]
    else:
        return subdirectories[default_index - 1]

# ...

selected_subdirectory = get_user_input(subdirectories, default_choice=subdirectories.index(most_recently_modified) + 1)