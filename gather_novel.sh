mkdir -p images/$2
scp -o 'ProxyCommand ssh lbethune@ssh.ens-lyon.fr -W %h:%p' lbethune@$1.cbp.ens-lyon.fr:/local/lbethune/Potion/images/$2/novel.plk images/$2/