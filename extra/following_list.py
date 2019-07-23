import pickle

from utils import cache_path, load_data


def main():
    print('generate cache for following_list ...')
    seens = load_data('seens')
    user_info = load_data('users.json')

    following_list = {user: set(user_info[user]['following_list']) if user in user_info else set() for user in
                      seens.keys()}
    with open(cache_path('following_list.pickle'), 'wb') as f:
        pickle.dump(following_list, f)


if __name__ == '__main__':
    main()
