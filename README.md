# S_13_Reverse_Engineering_SMOllm135M


In this project, we are trying to reverse engineer SmoLlm2-135M model and train the synthesized model for 5000 steps with generated text and checkpoints for every 500 steps. After 5000 steps, the model has to be run for another 50 steps.

SmoLlm-135M yaml file is avaialble here "https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt/blob/main/135M/final/config.yaml"

Here's a high level view of the model:

Parameters: ~135M Attention Heads: 9 (with 3 key-value heads) Activation Function: SiLU (Swish) Vocab Size: 49,152 Sequence Length: 2,048 and Grouped Query Attention.

The original model is trained on Cosmopedia-v2. But, the dataset is too huge with 28 billion tokens. While, it is easier to use online training and train the model, it would take six hours with sequence_length=2048. My Colab's A100 GPU is supporting only 750 tokens at max. With 750 tokens, it would take 2.75 times more time making it sixteen hours and this project we are just trying to understand the model behaivour for 5000 steps. 5000 steps with 2048 sequence length can capture roughly the same context as 13650 steps with 750 sequence length, but that's quite a bit of stretch of simplification we are considering, since, the shorter sequence length means the model is seeing more frequent context truncation. However, if the next batch naturally follows from the previous, the model still retains continuity in learning. But, for the sake of brevity, let's continue with 750 sequence length.

Here's the model that I replicated:

Parameter:134.6M Attention Heads: 9 (with 3 key-value heads) Tokenizer:cosmo2-tokenizer Activation Function SWILU Vocab Size: 49,152 Sequence Length: 750

Few sample generated texts with losses
*****************************************************************************************************************************************************
Completed interval 4/10

Input Sequence:

KING RICHARD II:
Thanks, noble peer;
The cheapest of us is ten groats too dear.
What art thou? and how comest thou hither,
Where no man never comes but that sad dog
That brings me food to make misfortune live?

Groom:
I was a poor groom of thy stable, king,
When thou wert king; who, travelling towards York,
With much ado at length have gotten leave
To look upon my sometimes royal master's face.
O, how it yearn'd my heart when I beheld
In London streets, that coronation-day,
When Bolingbroke rode on roan Barbary,
That horse that thou so often hast bestrid,
That horse that I so carefully have dress'd!

KING RICHARD II:
Rode he on Barbary? Tell me, gentle friend,
How went he under him?

Groom:
So proudly as if he disdain'd the ground.

KING RICHARD II:
So proud that Bolingbroke was on his back!
That jade hath eat bread from my royal hand;
This hand hath made him proud with clapping him.
Would he not stumble? would he not fall down,
Since pride must have a fall, and break the neck
Of that proud man that did usurp his back?
Forgiveness, horse! why do I rail on
--------------------------------------------------

Generated Continuation:
 thee
To be noble father died only nor it?

GLOUCESTER:
That whose easiest shall beheld!

KING RICHARD II:
We need not howl: let us bear it from heaven?

KING RICHARD II:
Then art not my head about me,
And at thy head and soft and myself
Shall lose yewks my rest,
Have taken hopes well-inius when he try o'
**************************************************
Steps in current interval: 100%|████████████████████████████████████████████████████| 500/500 [03:22<00:00,  2.46it/s, Average Loss=0.0142]

Completed interval 5/10

Input Sequence:

A hell-hound that doth hunt us all to death:
That dog, that had his teeth before his eyes,
To worry lambs and lap their gentle blood,
That foul defacer of God's handiwork,
That excellent grand tyrant of the earth,
That reigns in galled eyes of weeping souls,
Thy womb let loose, to chase us to our graves.
O upright, just, and true-disposing God,
How do I thank thee, that this carnal cur
Preys on the issue of his mother's body,
And makes her pew-fellow with others' moan!

DUCHESS OF YORK:
O Harry's wife, triumph not in my woes!
God witness with me, I have wept for thine.

QUEEN MARGARET:
Bear with me; I am hungry for revenge,
And now I cloy me with beholding it.
Thy Edward he is dead, that stabb'd my Edward:
Thy other Edward dead, to quit my Edward;
Young York he is but boot, because both they
Match not the high perfection of my loss:
Thy Clarence he is dead that kill'd my Edward;
And the beholders of this tragic play,
The adulterate Hastings, Rivers, Vaughan, Grey,
Untimely smother'd in their dusky graves.
Richard yet lives, hell's black intelligencer,
Only reserved their factor, to buy souls
And send them thither: but at hand, at hand,
Ensues his piteous and unpitied end:
Earth gapes, hell burns, fiends roar, saints pray.
To have him suddenly convey'd away.
Cancel his bond of life, dear God, I prey,
That I may live
--------------------------------------------------

Generated Continuation:
 to cannot grievous into seest together,
Or if it till my natural king, and set forth
Are often damn no shadow.

GLOUCESTER:
Players, dead;, I but that stabbroke:
No, husband, it, are all down as good ladies,
Either yet above my followers I read ruin.

DUCHESS OF Ely promised me,
And therefore ha doubt? or else have it is spent

**************************************************
Steps in current interval: 100%|████████████████████████████████████████████████████| 500/500 [03:22<00:00,  2.46it/s, Average Loss=0.0072]

Completed interval 6/10

Input Sequence:

A worthy officer i' the war; but insolent,
O'ercome with pride, ambitious past all thinking,
Self-loving,--

SICINIUS:
And affecting one sole throne,
Without assistance.

MENENIUS:
I think not so.

SICINIUS:
We should by this, to all our lamentation,
If he had gone forth consul, found it so.

BRUTUS:
The gods have well prevented it, and Rome
Sits safe and still without him.

AEdile:
Worthy tribunes,
There is a slave, whom we have put in prison,
Reports, the Volsces with two several powers
Are enter'd in the Roman territories,
And with the deepest malice of the war
Destroy what lies before 'em.

MENENIUS:
'Tis Aufidius,
Who, hearing of our Marcius' banishment,
Thrusts forth his horns again into the world;
Which were inshell'd when Marcius stood for Rome,

SICINIUS:
Come, what talk you
Of Marcius?

BRUTUS:
Go see this rumourer whipp'd. It cannot be
The Volsces dare break with us.

MENENIUS:
Cannot be!
We have record that very well it can,
And three examples of the like have been
Within my age. But reason with the fellow,
Before you punish him, where he heard this,
Lest you shall chance to whip your information
And beat the messenger who bids beware
Of what is to be dreaded.

SICINIUS:
Tell not me:
I know this cannot be.

BRUTUS:
Not possible.

Messenger:
The nobles in great earnestness are going
All to the senate-house: some news is come
That turns their countenances.

SICIN
--------------------------------------------------

Generated Continuation:
IUS:
He is, sir.

CORIOLANUS:
Pray, I know how assured
As 'tis by proud man.
SICINIUS: lovest.

MENENIUS:
Pray, I'll not prevent too o'rt a-house!

MENENIUS:
You shouldst not thyself.
If the poor curbs our further,
As prisoners, is und
**************************************************
Steps in current interval: 100%|████████████████████████████████████████████████████| 500/500 [03:22<00:00,  2.46it/s, Average Loss=0.0052]

Completed interval 7/10

Input Sequence:

Hast thou beheld a fresher gentlewoman?
Such war of white and red within her cheeks!
What stars do spangle heaven with such beauty,
As those two eyes become that heavenly face?
Fair lovely maid, once more good day to thee.
Sweet Kate, embrace her for her beauty's sake.

HORTENSIO:
A' will make the man mad, to make a woman of him.

KATHARINA:
Young budding virgin, fair and fresh and sweet,
Whither away, or where is thy abode?
Happy the parents of so fair a child;
Happier the man, whom favourable stars
Allot thee for his lovely bed-fellow!

PETRUCHIO:
Why, how now, Kate! I hope thou art not mad:
This is a man, old, wrinkled, faded, wither'd,
And not a maiden, as thou say'st he is.

KATHARINA:
Pardon, old father, my mistaking eyes,
That have been so bedazzled with the sun
That everything I look on seemeth green:
Now I perceive thou art a reverend father;
Pardon, I pray thee, for my mad mistaking.

PETRUCHIO:
Do, good old grandsire; and withal make known
Which way thou travellest: if along with us,
We shall
--------------------------------------------------

Generated Continuation:
 be cross it by.

GREMIO:
He straight shall any for over at hell burns?

PETRUCHIO:
Young cousin, my lord; I hence thou go.

NORTHUMBERLAND:
Ay, as you show more!

PARIS:
Away!

AUTOLYCUS:
P knows they do by your brothers.

PETRUCHIO:
Pray, why is it,
**************************************************
Steps in current interval: 100%|████████████████████████████████████████████████████| 500/500 [03:22<00:00,  2.46it/s, Average Loss=0.0041]

Completed interval 8/10

Input Sequence:

to. Lord Angelo dukes it well in his absence; he
puts transgression to 't.

DUKE VINCENTIO:
He does well in 't.

LUCIO:
A little more lenity to lechery would do no harm in
him: something too crabbed that way, friar.

DUKE VINCENTIO:
It is too general a vice, and severity must cure it.

LUCIO:
Yes, in good sooth, the vice is of a great kindred;
it is well allied: but it is impossible to extirp
it quite, friar, till eating and drinking be put
down. They say this Angelo was not made by man and
woman after this downright way of creation: is it
true, think you?

DUKE VINCENTIO:
How should he be made, then?

LUCIO:
Some report a sea-maid spawned him; some, that he
was begot between two stock-fishes. But it is
certain that when he makes water his urine is
congealed ice; that I know to be true: and he is a
motion generative; that's infallible.

DUKE VINCENTIO:
You are pleasant, sir, and speak apace.

LUCIO:
Why, what a ruthless thing is this in him, for the
rebellion of a codpiece to take away the life of a
man! Would the duke that is absent have done this?
Ere he would have hanged a man for the getting a
hundred bastards, he would have paid for the nursing
a thousand: he had some feeling of the sport: he

--------------------------------------------------

Generated Continuation:
k'd confound fellow; the duke was son, by
To tamed to wedded of doth he not,
By all the mean it of a prophetess
As vain: a and that answer it.

LUCESTER:
A very much.

First Gentleman:
Now, I talk of your dream,
Of his nature?Why do rage, when you please you in our justice
From your mistress that is as much again,
Upon
**************************************************
Steps in current interval: 100%|████████████████████████████████████████████████████| 500/500 [03:22<00:00,  2.46it/s, Average Loss=0.0035]

Completed interval 9/10

Input Sequence:

Whom for this time we pardon. We enjoin thee,
As thou art liege-man to us, that thou carry
This female bastard hence and that thou bear it
To some remote and desert place quite out
Of our dominions, and that there thou leave it,
Without more mercy, to its own protection
And favour of the climate. As by strange fortune
It came to us, I do in justice charge thee,
On thy soul's peril and thy body's torture,
That thou commend it strangely to some place
Where chance may nurse or end it. Take it up.

ANTIGONUS:
I swear to do this, though a present death
Had been more merciful. Come on, poor babe:
Some powerful spirit instruct the kites and ravens
To be thy nurses! Wolves and bears, they say
Casting their savageness aside have done
Like offices of pity. Sir, be prosperous
In more than this deed does require! And blessing
Against this cruelty fight on thy side,
Poor thing, condemn'd to loss!

LEONTES:
No, I'll not rear
Another's issue.

Servant:
Please your highness, posts
From those you sent to the oracle are come
An hour since: Cleomenes
--------------------------------------------------

Generated Continuation:
 and be
Of celebrationigh!

MENENIUS:
Let me, AUMBERLAND:
No, O, sorrow's no more as he till it.

PROSPERO:
As three months so
As he?

MENENIUS:
The bade me. I'll not think;
Where is a' the wayward:

SICINIUS:

In time, backless be yet.
