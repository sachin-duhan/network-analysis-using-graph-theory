import os
import instaloader

data_dir = "data" # direction to store data
my_username = 'enter_your_username_here'
my_password = 'enter_your_password_here'


def get_friends(username):
    # find users that are following and are followed by the given user

    profile = instaloader.Profile.from_username(loader.context, username)
    followers = profile.get_followers()
    
    profile = instaloader.Profile.from_username(loader.context, username)
    followees = profile.get_followees()
    
    friends = set(followers).intersection(followees)
    friends_username = [friend.username for friend in friends]
    
    f = open(os.path.join(data_dir, profile.username + '.txt'), 'w')
    f.write('\n'.join(friends_username))
    f.close()



loader = instaloader.Instaloader()
loader.login(my_username, my_password)

# find my friends
my_profile = instaloader.Profile.from_username(loader.context, my_username)
my_followers = my_profile.get_followers()
my_followees = my_profile.get_followees()
my_friends = set(my_followers).intersection(my_followees)
my_friends_username = [friend.username for friend in my_friends]

# f = open(os.path.join(data_dir, my_username + '.txt'), 'w')
# f.write('\n'.join(my_friends_username))
# f.close()

# save friends data 
for username in my_friends_username:
    print(username)
    if not os.path.isfile(os.path.join(data_dir, username + '.txt')):
        get_friends(username)